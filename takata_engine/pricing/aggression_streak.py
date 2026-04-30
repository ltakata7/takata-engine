"""Aggression streak tracker — N consecutive same-direction agg deltas.

When 5+ bars in a row print net positive aggression (buyer-initiated trades
dominating), it's a thrust signal: real participants are pressing in one
direction with conviction. Pair it with confluence on the trend side and
it becomes a high-quality entry.

Streaks invert quickly when the move exhausts — when a streak that's been
running for 5+ bars suddenly gets a counter-direction bar with notably
larger volume, that's the "thrust → exhaustion" signature, often the local top.

Two states surfaced:
- `direction`: "bull" | "bear" | "neutral" (current streak direction)
- `length`: how many consecutive bars have agreed
- `exhaustion`: True when a recent streak just reversed on heavier volume
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, Optional

# A streak must have ≥ this many bars to be considered "real."
MIN_STREAK_LENGTH = 3

# How recent the exhaustion reversal can be before we stop reporting it (seconds).
EXHAUSTION_TTL_S = 300.0

# Min agg balance magnitude per bar to count as "directional."
MIN_BAR_AGG = 50.0

# How many bars to keep history for. 60 = ~5h on 5min bars.
HISTORY_LEN = 60


@dataclass
class StreakBar:
    timestamp: float
    price: float
    agg_balance: float
    volume: float


@dataclass
class StreakState:
    direction: str = "neutral"   # "bull" | "bear" | "neutral"
    length: int = 0
    started_at: float = 0.0
    started_price: float = 0.0
    cumulative_agg: float = 0.0   # sum of bar aggs over this streak
    exhaustion_at: float = 0.0    # ts of last exhaustion reversal (0 if none)
    exhaustion_direction: str = "none"  # the direction the streak HAD before reversing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "length": self.length,
            "started_at": self.started_at,
            "started_iso": datetime.fromtimestamp(self.started_at).isoformat() if self.started_at else None,
            "started_price": round(self.started_price, 2),
            "cumulative_agg": round(self.cumulative_agg, 0),
            "exhaustion_at": self.exhaustion_at,
            "exhaustion_direction": self.exhaustion_direction,
        }


class AggressionStreakTracker:
    """Tracks consecutive same-direction aggression bars + exhaustion reversals.

    Usage:
        t = AggressionStreakTracker()
        t.update(price, agg_balance, volume)
        sig = t.signal()
    """

    def __init__(self) -> None:
        self._state = StreakState()
        self._history: Deque[StreakBar] = deque(maxlen=HISTORY_LEN)

    def update(
        self,
        price: float,
        agg_balance: float,
        volume: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> None:
        if price <= 0:
            return
        ts = timestamp if timestamp is not None else datetime.now().timestamp()

        # Classify this bar's direction.
        if abs(agg_balance) < MIN_BAR_AGG:
            bar_dir = "neutral"
        elif agg_balance > 0:
            bar_dir = "bull"
        else:
            bar_dir = "bear"

        # Streak continuation vs reversal.
        st = self._state
        prev_dir = st.direction
        prev_len = st.length

        if bar_dir == "neutral":
            # Doesn't extend or break a streak — just hold state.
            pass
        elif bar_dir == prev_dir:
            st.length += 1
            st.cumulative_agg += agg_balance
        else:
            # Reversal — check for exhaustion (prior streak was real and this
            # counter-bar has bigger volume than the recent streak average).
            if prev_len >= MIN_STREAK_LENGTH and self._history:
                recent_streak_vols = [
                    b.volume for b in list(self._history)[-prev_len:]
                    if b.volume > 0
                ]
                if recent_streak_vols and volume > 0:
                    avg_streak_vol = sum(recent_streak_vols) / len(recent_streak_vols)
                    if volume > avg_streak_vol * 1.3:
                        st.exhaustion_at = ts
                        st.exhaustion_direction = prev_dir
            # Start fresh streak.
            st.direction = bar_dir
            st.length = 1
            st.started_at = ts
            st.started_price = price
            st.cumulative_agg = agg_balance

        # If we transitioned out of neutral, set started_at on first directional bar.
        if prev_dir == "neutral" and bar_dir != "neutral":
            st.direction = bar_dir
            st.length = 1
            st.started_at = ts
            st.started_price = price
            st.cumulative_agg = agg_balance

        self._history.append(StreakBar(
            timestamp=ts, price=price, agg_balance=agg_balance, volume=volume,
        ))

    def signal(self) -> Dict[str, Any]:
        """Return the streak signal for scanner consumption.

        Includes:
        - active: whether a "real" streak (>= MIN_STREAK_LENGTH) is live
        - direction: current streak direction
        - length: bars in current streak
        - exhaustion_active: whether a recent reversal just printed exhaustion
        - score_adjustment: small confluence nudge for trend agreement
        """
        st = self._state
        active = st.length >= MIN_STREAK_LENGTH and st.direction != "neutral"
        now = datetime.now().timestamp()
        exhaustion_active = (
            st.exhaustion_at > 0 and
            (now - st.exhaustion_at) < EXHAUSTION_TTL_S
        )

        score_adjustment = 0.0
        reasons = []
        if active:
            # Longer streaks earn more — diminishing returns past 7.
            score_adjustment = min(0.15, 0.04 * st.length)
            # Static reason name for classifier matching; length carried in data dict.
            reasons.append(f"agg_streak_{st.direction}")
        if exhaustion_active:
            reasons.append(f"agg_exhaustion_{st.exhaustion_direction}")

        return {
            "active": active,                         # True only at length ≥ MIN_STREAK_LENGTH
            "direction": st.direction,                # actual current streak direction
            "length": st.length,
            "cumulative_agg": round(st.cumulative_agg, 0),
            "exhaustion_active": exhaustion_active,
            "exhaustion_direction": st.exhaustion_direction if exhaustion_active else "none",
            "score_adjustment": score_adjustment,
            "reasons": reasons,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self._state.to_dict(),
            "bars_seen": len(self._history),
            "signal": self.signal(),
        }


_singletons: Dict[str, AggressionStreakTracker] = {}


def get_aggression_streak_tracker(instrument: str = "WDO") -> AggressionStreakTracker:
    """Per-instrument singleton so MES and WDO don't contaminate each other."""
    if instrument not in _singletons:
        _singletons[instrument] = AggressionStreakTracker()
    return _singletons[instrument]
