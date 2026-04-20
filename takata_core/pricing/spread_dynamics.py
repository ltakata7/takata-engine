"""Bid/Ask spread dynamics tracker for WDO fast-timeframe trading.

Tracks the evolution of the best-bid / best-ask spread over a rolling window
and flags three high-information events:

**Why spread dynamics matter for WDO 3-5min trading:**
- Spread compression (tight spread relative to session) = liquidity present,
  institutional makers working both sides → high-probability breakouts
- Spread widening (wider than session norm) = liquidity withdrawal before
  a move → position sizing should shrink, directional signals carry weight
- Spread rejection (sudden spike then snap-back) = stop-hunt signature →
  contrarian reversal setup (the "wick" everyone sees on a 1-min chart)

Spread behavior leads price ~5-30 seconds in most markets — makers widen
before they step away, and they tighten when they're comfortable. That
short lead is exactly what a 3-5 min trader needs.

Signal types generated:
- ``spread_compressed``    — spread in bottom 20% of session range
- ``spread_compressed_primed`` — compressed AND price near ORB/VWAP edge
- ``spread_widening``      — spread spiking 2σ above rolling mean
- ``spread_rejection``     — spread spiked and immediately compressed back
- ``spread_dead``          — spread stable and below median = quiet regime
- ``spread_trending_wide`` — spread rolling mean rising over time

Call ``update(bid, ask)`` every scan cycle.
Call ``signal()`` for the current spread analysis.
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

# Keep ~5 min of spread history at 2s tick rate = 150 samples
SPREAD_WINDOW = 150

# Short burst window for rejection detection (~20 samples ≈ 40s)
REJECTION_WINDOW = 20

# Thresholds
COMPRESSION_PCT = 20.0   # spread below this percentile = compressed
WIDENING_PCT = 80.0      # spread above this percentile = widening
REJECTION_SPIKE_RATIO = 2.5  # spike must be 2.5x the recent mean


@dataclass
class SpreadState:
    """Spread history buffers."""
    date: str = ""
    history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=SPREAD_WINDOW)
    )  # (ts, spread)
    # Track session min/max for context
    session_min: float = float("inf")
    session_max: float = 0.0

    def reset(self, today: str) -> None:
        self.date = today
        self.history = deque(maxlen=SPREAD_WINDOW)
        self.session_min = float("inf")
        self.session_max = 0.0


class SpreadDynamicsTracker:
    """Tracks bid/ask spread evolution and flags liquidity regime shifts."""

    def __init__(self) -> None:
        self._state = SpreadState()

    def update(self, bid: float, ask: float) -> None:
        """Record a new spread reading.

        Parameters
        ----------
        bid : float
            Best bid price.
        ask : float
            Best ask price.
        """
        if bid <= 0 or ask <= 0 or ask < bid:
            return
        spread = ask - bid
        # Reject unreasonable spreads (sometimes bridge snapshots cross)
        if spread > 20.0:
            return

        now = datetime.now(BRT)
        today_str = now.strftime("%Y-%m-%d")
        if self._state.date != today_str:
            self._state.reset(today_str)

        self._state.history.append((now.timestamp(), spread))
        if spread > 0:
            if spread < self._state.session_min:
                self._state.session_min = spread
            if spread > self._state.session_max:
                self._state.session_max = spread

    def signal(self) -> Dict[str, Any]:
        """Generate spread-dynamics signal dict."""
        result: Dict[str, Any] = {
            "current_spread": 0.0,
            "mean_spread": 0.0,
            "median_spread": 0.0,
            "session_min": 0.0,
            "session_max": 0.0,
            "percentile": 0.0,
            "regime": "insufficient_data",
            "signal": "neutral",
            "score_adjustment": 0.0,
            "reasons": [],
        }
        reasons: List[str] = result["reasons"]
        adj = 0.0

        history = list(self._state.history)
        if len(history) < 20:
            return result

        spreads = [s for _, s in history]
        current = spreads[-1]
        mean = sum(spreads) / len(spreads)
        sorted_sp = sorted(spreads)
        median = sorted_sp[len(sorted_sp) // 2]
        # Rank-based percentile of current spread
        rank = sum(1 for s in sorted_sp if s <= current)
        pct = rank / len(sorted_sp) * 100.0

        result["current_spread"] = round(current, 2)
        result["mean_spread"] = round(mean, 2)
        result["median_spread"] = round(median, 2)
        result["session_min"] = round(
            self._state.session_min if self._state.session_min != float("inf") else 0, 2
        )
        result["session_max"] = round(self._state.session_max, 2)
        result["percentile"] = round(pct, 1)

        # Rolling std for z-score
        var = sum((s - mean) ** 2 for s in spreads) / max(len(spreads) - 1, 1)
        std = var ** 0.5 if var > 0 else 0.0
        z = (current - mean) / std if std > 0 else 0.0
        result["z_score"] = round(z, 2)

        # ── 1. Spread compression ──
        if pct <= COMPRESSION_PCT:
            result["regime"] = "compressed"
            reasons.append("spread_compressed")
            adj += 0.05

        # ── 2. Spread widening ──
        elif pct >= WIDENING_PCT:
            result["regime"] = "widening"
            reasons.append("spread_widening")
            adj += 0.04  # scanner interprets as caution — direction TBD
            result["signal"] = "liquidity_retreat"

        elif pct <= 50:
            result["regime"] = "below_median"
        else:
            result["regime"] = "above_median"

        # ── 3. Spread rejection — spike that instantly compressed ──
        if len(history) >= REJECTION_WINDOW + 1:
            recent = spreads[-REJECTION_WINDOW:]
            recent_mean = sum(recent[:-1]) / (len(recent) - 1)  # exclude latest
            peak = max(recent)
            # Spike happened recently but current is back down
            if recent_mean > 0 and peak / recent_mean > REJECTION_SPIKE_RATIO:
                if current <= recent_mean * 1.3:  # reverted within 30% of mean
                    reasons.append("spread_rejection")
                    adj += 0.08  # stop-hunt reversal is high-value
                    result["signal"] = "rejection_reversal"

        # ── 4. Dead tape — stable, below median ──
        if std > 0 and std / mean < 0.10 and current < median:
            reasons.append("spread_dead")
            adj += 0.02  # quiet regime, favors breakouts once vol returns

        # ── 5. Trending wide — rolling mean rising vs first half ──
        if len(history) >= 60:
            first_half = spreads[: len(spreads) // 2]
            second_half = spreads[len(spreads) // 2 :]
            fh_mean = sum(first_half) / len(first_half)
            sh_mean = sum(second_half) / len(second_half)
            if fh_mean > 0 and sh_mean / fh_mean > 1.25:
                reasons.append("spread_trending_wide")
                adj += 0.03
                result["signal"] = "liquidity_deteriorating"

        # ── 6. Compression primed — tight spread flag for consumer ──
        # The consumer (scanner) checks if we're also near ORB/VWAP edge
        # and will add "spread_compressed_primed" itself. We flag readiness.
        if pct <= COMPRESSION_PCT and std / mean < 0.15 if mean > 0 else False:
            result["compression_primed"] = True
        else:
            result["compression_primed"] = False

        result["score_adjustment"] = round(adj, 3)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Dump current state for API/frontend."""
        history = list(self._state.history)
        spreads = [s for _, s in history]
        mean = sum(spreads) / len(spreads) if spreads else 0.0
        return {
            "date": self._state.date,
            "current_spread": round(spreads[-1], 2) if spreads else 0,
            "mean_spread": round(mean, 2),
            "session_min": round(
                self._state.session_min if self._state.session_min != float("inf") else 0, 2
            ),
            "session_max": round(self._state.session_max, 2),
            "history_depth": len(spreads),
        }


# Singleton accessor
_spread_tracker_instance: Optional[SpreadDynamicsTracker] = None


def get_spread_tracker() -> SpreadDynamicsTracker:
    """Get the singleton spread-dynamics tracker."""
    global _spread_tracker_instance
    if _spread_tracker_instance is None:
        _spread_tracker_instance = SpreadDynamicsTracker()
    return _spread_tracker_instance
