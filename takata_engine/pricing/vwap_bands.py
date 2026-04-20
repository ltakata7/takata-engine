"""VWAP with σ-bands tracker — session VWAP plus ±1σ / ±2σ bands.

Tracks price stretch relative to session VWAP using volume-weighted
standard deviation bands. These bands are the single most-watched
intraday reference for Brazilian prop desks after the PTAX windows.

**Why VWAP bands matter for WDO 3-5min trading:**
- Price oscillates around VWAP; 2σ stretches usually mean-revert
- Band reclaim after rejection = momentum continuation signal
- Distance from VWAP (in σ units) = position-sizing input
- Walk-the-band behavior (riding 1σ band) = strong trend day

Signal types generated:
- ``vwap_stretch_long``  — price at or below -2σ = buy the dip
- ``vwap_stretch_short`` — price at or above +2σ = fade the rally
- ``vwap_reclaim_long``  — price crossed back up through VWAP from below
- ``vwap_reclaim_short`` — price crossed back down through VWAP from above
- ``vwap_band_ride_bull`` — price walking the +1σ band = strong uptrend
- ``vwap_band_ride_bear`` — price walking the -1σ band = strong downtrend
- ``vwap_1sigma_break_long``  — breakout past +1σ with momentum
- ``vwap_1sigma_break_short`` — breakout past -1σ with momentum

Implementation: Welford's online algorithm for running variance (stable,
no need to keep all ticks in memory).

Call ``update()`` every scan cycle with price + volume delta.
Call ``signal()`` to get the current VWAP band signal dict.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)

BRT = pytz.timezone("America/Sao_Paulo")

SESSION_OPEN_H = 9
SESSION_OPEN_M = 0

# How many recent (price, σ-distance) points to keep for band-ride detection
BAND_RIDE_WINDOW = 30

# Minimum ticks needed before bands are meaningful
MIN_TICKS_FOR_BANDS = 20


@dataclass
class VWAPState:
    """Session VWAP running state — resets each day."""
    date: str = ""
    cum_pv: float = 0.0           # Σ price × volume
    cum_v: float = 0.0            # Σ volume
    # Welford variance components (volume-weighted)
    cum_pv_sq_diff: float = 0.0   # Σ v × (price - mean)²
    vwap: float = 0.0
    variance: float = 0.0
    n_ticks: int = 0
    # Recent σ-distance history for band-ride detection
    sigma_history: Deque[float] = field(default_factory=lambda: deque(maxlen=BAND_RIDE_WINDOW))
    # Track VWAP crosses for reclaim signals
    prev_side: str = "unknown"    # "above", "below", "unknown"

    def reset(self, today: str) -> None:
        self.date = today
        self.cum_pv = 0.0
        self.cum_v = 0.0
        self.cum_pv_sq_diff = 0.0
        self.vwap = 0.0
        self.variance = 0.0
        self.n_ticks = 0
        self.sigma_history = deque(maxlen=BAND_RIDE_WINDOW)
        self.prev_side = "unknown"


class VWAPBandsTracker:
    """Tracks session VWAP and ±1σ / ±2σ bands.

    The bridge already publishes a session VWAP, but it doesn't expose the
    dispersion bands. This tracker reconstructs both from tick data.
    """

    def __init__(self) -> None:
        self._state = VWAPState()
        self._last_cross_ts: float = 0.0
        self._cross_cooldown_s: float = 60.0  # don't double-fire reclaim signals

    def update(self, price: float, volume_delta: float) -> None:
        """Update VWAP + variance running totals.

        Parameters
        ----------
        price : float
            Current trade price.
        volume_delta : float
            Volume traded since last update (contracts).
        """
        if price <= 0 or volume_delta <= 0:
            # We still want to compare price to current vwap for cross detection
            self._update_cross(price)
            return

        now = datetime.now(BRT)
        today_str = now.strftime("%Y-%m-%d")

        # Day reset at session open
        if self._state.date != today_str:
            self._state.reset(today_str)

        mins_since_open = (now.hour - SESSION_OPEN_H) * 60 + (now.minute - SESSION_OPEN_M)
        if mins_since_open < 0:
            return  # pre-session

        # Welford-style volume-weighted variance update
        prev_cum_v = self._state.cum_v
        new_cum_v = prev_cum_v + volume_delta

        if prev_cum_v == 0:
            self._state.vwap = price
            self._state.cum_v = volume_delta
            self._state.cum_pv = price * volume_delta
            self._state.n_ticks = 1
            return

        delta = price - self._state.vwap
        new_vwap = self._state.vwap + (delta * volume_delta) / new_cum_v
        self._state.cum_pv_sq_diff += volume_delta * delta * (price - new_vwap)
        self._state.vwap = new_vwap
        self._state.cum_v = new_cum_v
        self._state.cum_pv = self._state.cum_pv + price * volume_delta
        self._state.variance = self._state.cum_pv_sq_diff / new_cum_v if new_cum_v > 0 else 0
        self._state.n_ticks += 1

        # Track σ-distance for band-ride detection
        sigma = math.sqrt(max(self._state.variance, 0))
        if sigma > 0:
            sigma_dist = (price - new_vwap) / sigma
            self._state.sigma_history.append(sigma_dist)

        self._update_cross(price)

    def _update_cross(self, price: float) -> None:
        """Detect VWAP crosses for reclaim signals."""
        if self._state.vwap <= 0:
            return
        current_side = "above" if price > self._state.vwap else "below"
        if self._state.prev_side == "unknown":
            self._state.prev_side = current_side
            return
        if current_side != self._state.prev_side:
            self._last_cross_ts = datetime.now().timestamp()
        self._state.prev_side = current_side

    def signal(self, current_price: float) -> Dict[str, Any]:
        """Generate VWAP-bands signal dict.

        Returns dict with vwap, bands, sigma distance, and reasons.
        """
        result: Dict[str, Any] = {
            "vwap": round(self._state.vwap, 1) if self._state.vwap else 0,
            "sigma": 0.0,
            "upper_1sigma": 0.0,
            "lower_1sigma": 0.0,
            "upper_2sigma": 0.0,
            "lower_2sigma": 0.0,
            "sigma_distance": 0.0,
            "signal": "neutral",
            "score_adjustment": 0.0,
            "reasons": [],
        }
        reasons: List[str] = result["reasons"]
        adj = 0.0

        if self._state.n_ticks < MIN_TICKS_FOR_BANDS or self._state.vwap <= 0:
            return result

        sigma = math.sqrt(max(self._state.variance, 0))
        if sigma <= 0:
            return result

        vwap = self._state.vwap
        result["sigma"] = round(sigma, 2)
        result["upper_1sigma"] = round(vwap + sigma, 1)
        result["lower_1sigma"] = round(vwap - sigma, 1)
        result["upper_2sigma"] = round(vwap + 2 * sigma, 1)
        result["lower_2sigma"] = round(vwap - 2 * sigma, 1)

        sigma_dist = (current_price - vwap) / sigma
        result["sigma_distance"] = round(sigma_dist, 2)

        # 1. 2σ stretch = mean reversion opportunity
        if sigma_dist <= -2.0:
            reasons.append("vwap_stretch_long")
            adj += 0.10
            result["signal"] = "stretched_oversold"
        elif sigma_dist >= 2.0:
            reasons.append("vwap_stretch_short")
            adj += 0.10
            result["signal"] = "stretched_overbought"

        # 2. Reclaim — crossed back through VWAP recently
        import time
        secs_since_cross = time.time() - self._last_cross_ts
        if 0 < secs_since_cross < self._cross_cooldown_s:
            if self._state.prev_side == "above":
                reasons.append("vwap_reclaim_long")
                adj += 0.08
            elif self._state.prev_side == "below":
                reasons.append("vwap_reclaim_short")
                adj += 0.08

        # 3. Band ride — last N ticks all on same side of 1σ band = trend day
        if len(self._state.sigma_history) >= BAND_RIDE_WINDOW:
            history = list(self._state.sigma_history)
            ride_up = sum(1 for s in history if s >= 1.0) / len(history)
            ride_dn = sum(1 for s in history if s <= -1.0) / len(history)
            if ride_up > 0.7:
                reasons.append("vwap_band_ride_bull")
                adj += 0.08  # strong uptrend continuation
                result["signal"] = "trending_up"
            elif ride_dn > 0.7:
                reasons.append("vwap_band_ride_bear")
                adj += 0.08
                result["signal"] = "trending_down"

        # 4. 1σ breakout — fresh move past ±1σ band
        if 1.0 <= sigma_dist < 2.0:
            reasons.append("vwap_1sigma_break_long")
            adj += 0.05
        elif -2.0 < sigma_dist <= -1.0:
            reasons.append("vwap_1sigma_break_short")
            adj += 0.05

        result["score_adjustment"] = round(adj, 3)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Dump current state for API/frontend."""
        sigma = math.sqrt(max(self._state.variance, 0))
        return {
            "date": self._state.date,
            "vwap": round(self._state.vwap, 1) if self._state.vwap else 0,
            "sigma": round(sigma, 2),
            "n_ticks": self._state.n_ticks,
            "cum_volume": round(self._state.cum_v, 0),
            "upper_1sigma": round(self._state.vwap + sigma, 1) if self._state.vwap else 0,
            "lower_1sigma": round(self._state.vwap - sigma, 1) if self._state.vwap else 0,
            "upper_2sigma": round(self._state.vwap + 2 * sigma, 1) if self._state.vwap else 0,
            "lower_2sigma": round(self._state.vwap - 2 * sigma, 1) if self._state.vwap else 0,
        }


# Singleton accessor
_vwap_bands_instance: Optional[VWAPBandsTracker] = None


def get_vwap_bands_tracker() -> VWAPBandsTracker:
    """Get the singleton VWAP bands tracker."""
    global _vwap_bands_instance
    if _vwap_bands_instance is None:
        _vwap_bands_instance = VWAPBandsTracker()
    return _vwap_bands_instance
