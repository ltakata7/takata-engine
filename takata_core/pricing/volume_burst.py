"""Volume burst tracker for WDO fast-timeframe trading.

Detects anomalous volume activity vs a rolling baseline. Volume bursts
are leading indicators of institutional participation — a sudden 3× spike
in per-minute contracts almost always precedes a directional move.

**Why volume bursts matter for WDO 3-5min trading:**
- Volume precedes price: institutions cross size before price moves far
- Burst at session high/low = climax = reversal
- Burst mid-range + trend = continuation
- Dead tape (sub-median volume) = price drift, high risk of whipsaw
- Sustained above-average volume = committed trend day

The tracker keeps 1-min volume buckets for the last 20 minutes and
compares the most recent bucket to the rolling median + MAD.

Signal types generated:
- ``vol_burst_bull``  — volume spike + price up = institutional entry
- ``vol_burst_bear``  — volume spike + price down = institutional exit
- ``vol_climax_high`` — burst near session high = reversal setup
- ``vol_climax_low``  — burst near session low = reversal setup
- ``vol_sustained_high`` — 3+ consecutive 1-min buckets above 1.5× median
- ``vol_dead_tape``   — current 1-min vol below 30th percentile = drift regime
- ``vol_expanding``   — rolling mean rising (session building momentum)

Call ``update(price, cum_volume)`` every scan cycle with cumulative session
volume. Call ``signal(current_price)`` for the analysis dict.
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

# Per-minute bucket history — keep last 20 minutes
MINUTE_BUCKETS = 20

# Minute bucket is built by aggregating ~30 ticks (scanner runs every ~2s)
BUCKET_SECONDS = 60

# Burst thresholds
BURST_RATIO = 2.5       # current vs rolling median
STRONG_BURST_RATIO = 4.0

# Dead-tape threshold (percentile)
DEAD_PCT = 30.0

# Climax proximity to session high/low (WDO points)
CLIMAX_PROXIMITY = 3.0


@dataclass
class MinuteBucket:
    """Per-minute aggregate."""
    start_ts: float
    end_ts: float
    volume: float = 0.0
    start_price: float = 0.0
    end_price: float = 0.0


@dataclass
class VolumeState:
    """Volume-burst tracker state."""
    date: str = ""
    prev_cum_volume: float = 0.0
    # Current in-progress bucket + closed buckets
    current_bucket: Optional[MinuteBucket] = None
    closed_buckets: Deque[MinuteBucket] = field(
        default_factory=lambda: deque(maxlen=MINUTE_BUCKETS)
    )
    session_high_price: float = 0.0
    session_low_price: float = float("inf")

    def reset(self, today: str) -> None:
        self.date = today
        self.prev_cum_volume = 0.0
        self.current_bucket = None
        self.closed_buckets = deque(maxlen=MINUTE_BUCKETS)
        self.session_high_price = 0.0
        self.session_low_price = float("inf")


class VolumeBurstTracker:
    """Tracks volume activity regime and flags bursts/climax/dead-tape."""

    def __init__(self) -> None:
        self._state = VolumeState()

    def update(self, price: float, cum_volume: float) -> None:
        """Update volume-bucket state.

        Parameters
        ----------
        price : float
            Current price.
        cum_volume : float
            Cumulative session volume from the bridge (contracts).
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

        # Derive incremental volume
        vol_delta = 0.0
        if self._state.prev_cum_volume > 0:
            vol_delta = max(cum_volume - self._state.prev_cum_volume, 0.0)
        self._state.prev_cum_volume = cum_volume

        # Track session extremes
        if price > self._state.session_high_price:
            self._state.session_high_price = price
        if self._state.session_low_price == float("inf") or price < self._state.session_low_price:
            self._state.session_low_price = price

        ts = now.timestamp()
        # Create first bucket
        if self._state.current_bucket is None:
            self._state.current_bucket = MinuteBucket(
                start_ts=ts, end_ts=ts + BUCKET_SECONDS,
                volume=vol_delta, start_price=price, end_price=price,
            )
            return

        bucket = self._state.current_bucket
        # Close the bucket if time has rolled over
        if ts >= bucket.end_ts:
            self._state.closed_buckets.append(bucket)
            self._state.current_bucket = MinuteBucket(
                start_ts=ts, end_ts=ts + BUCKET_SECONDS,
                volume=vol_delta, start_price=price, end_price=price,
            )
        else:
            bucket.volume += vol_delta
            bucket.end_price = price

    def signal(self, current_price: float) -> Dict[str, Any]:
        """Generate volume-burst signal dict."""
        result: Dict[str, Any] = {
            "current_bucket_volume": 0.0,
            "median_volume": 0.0,
            "burst_ratio": 0.0,
            "regime": "insufficient_data",
            "signal": "neutral",
            "score_adjustment": 0.0,
            "reasons": [],
        }
        reasons: List[str] = result["reasons"]
        adj = 0.0

        closed = list(self._state.closed_buckets)
        if len(closed) < 5 or self._state.current_bucket is None:
            return result

        current_vol = self._state.current_bucket.volume
        result["current_bucket_volume"] = round(current_vol, 0)
        result["buckets_closed"] = len(closed)

        # Baseline = median of closed buckets (robust to outliers)
        volumes = sorted(b.volume for b in closed)
        median = volumes[len(volumes) // 2]
        result["median_volume"] = round(median, 0)

        # Rolling mean & rising check
        first_half = volumes[: len(volumes) // 2]
        second_half = volumes[len(volumes) // 2 :]
        fh_mean = sum(first_half) / len(first_half) if first_half else 0
        sh_mean = sum(second_half) / len(second_half) if second_half else 0

        if median <= 0:
            return result

        ratio = current_vol / median if median > 0 else 0.0
        result["burst_ratio"] = round(ratio, 2)

        # ── 1. Burst detection (direction-aware) ──
        bucket = self._state.current_bucket
        price_move = bucket.end_price - bucket.start_price
        if ratio >= BURST_RATIO:
            result["regime"] = "burst"
            if price_move > 0:
                reasons.append("vol_burst_bull")
                adj += 0.08
                result["signal"] = "burst_up"
            elif price_move < 0:
                reasons.append("vol_burst_bear")
                adj += 0.08
                result["signal"] = "burst_down"
            if ratio >= STRONG_BURST_RATIO:
                # Stronger bursts get an additional boost
                adj += 0.04

        # ── 2. Climax — burst near session extremes ──
        near_high = abs(current_price - self._state.session_high_price) <= CLIMAX_PROXIMITY
        near_low = (
            self._state.session_low_price != float("inf")
            and abs(current_price - self._state.session_low_price) <= CLIMAX_PROXIMITY
        )
        if ratio >= BURST_RATIO and near_high:
            reasons.append("vol_climax_high")
            adj += 0.06  # reversal setup — upside exhaustion
            result["signal"] = "climax_top"
        elif ratio >= BURST_RATIO and near_low:
            reasons.append("vol_climax_low")
            adj += 0.06
            result["signal"] = "climax_bottom"

        # ── 3. Sustained activity — last 3 buckets all >1.5× median ──
        if len(closed) >= 3:
            last3 = list(closed)[-3:]
            if all(b.volume > median * 1.5 for b in last3):
                reasons.append("vol_sustained_high")
                adj += 0.05
                if result["regime"] == "insufficient_data":
                    result["regime"] = "sustained_high"

        # ── 4. Dead tape ──
        rank = sum(1 for v in volumes if v <= current_vol)
        pct = rank / len(volumes) * 100.0
        result["percentile"] = round(pct, 1)
        if pct <= DEAD_PCT:
            reasons.append("vol_dead_tape")
            adj -= 0.02  # discourage signals in dead tape
            if result["regime"] == "insufficient_data":
                result["regime"] = "dead_tape"

        # ── 5. Expanding — rolling mean rising ──
        if fh_mean > 0 and sh_mean / fh_mean > 1.3:
            reasons.append("vol_expanding")
            adj += 0.03

        result["score_adjustment"] = round(adj, 3)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Dump current state for API/frontend."""
        closed = list(self._state.closed_buckets)
        median = 0.0
        if closed:
            volumes = sorted(b.volume for b in closed)
            median = volumes[len(volumes) // 2]
        return {
            "date": self._state.date,
            "buckets_closed": len(closed),
            "current_bucket_volume": round(
                self._state.current_bucket.volume if self._state.current_bucket else 0, 0
            ),
            "median_bucket_volume": round(median, 0),
            "session_high_price": round(self._state.session_high_price, 1),
            "session_low_price": round(
                self._state.session_low_price
                if self._state.session_low_price != float("inf")
                else 0,
                1,
            ),
        }


# Singleton accessor
_volume_burst_instance: Optional[VolumeBurstTracker] = None


def get_volume_burst_tracker() -> VolumeBurstTracker:
    """Get the singleton volume-burst tracker."""
    global _volume_burst_instance
    if _volume_burst_instance is None:
        _volume_burst_instance = VolumeBurstTracker()
    return _volume_burst_instance
