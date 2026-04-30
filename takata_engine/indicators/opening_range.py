"""Opening Range — first N minutes of a session.

The opening range (OR) is the high/low established in the first N minutes
of the session. After OR locks at minute N, traders watch for:
- **Breakout above OR-high** with volume → continuation long
- **Breakdown below OR-low** with volume → continuation short
- **Failed breakout** (break + reclaim) → fade in opposite direction
- **Inside the range** → wait, often chop until breakout

Standard windows: 15 min (aggressive), 30 min (most common), 60 min (slow start).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional

import pandas as pd


@dataclass
class OpeningRange:
    """Computed opening range for a session.

    Attributes
    ----------
    high : float
        Highest price during the OR window.
    low : float
        Lowest price during the OR window.
    open : float
        First trade price of the session.
    locked_at : pd.Timestamp | None
        When the OR was finalized. None if still forming.
    bars_used : int
        How many bars contributed to the OR.
    """
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    locked_at: Optional[pd.Timestamp] = None
    bars_used: int = 0

    @property
    def is_locked(self) -> bool:
        return self.locked_at is not None

    @property
    def width(self) -> float:
        return max(0.0, self.high - self.low)

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2 if self.high > 0 and self.low > 0 else 0.0

    def state(self, price: float) -> str:
        """Where is `price` relative to the locked OR?

        Returns one of: "forming", "above", "below", "inside".
        """
        if not self.is_locked:
            return "forming"
        if price > self.high:
            return "above"
        if price < self.low:
            return "below"
        return "inside"

    def to_dict(self) -> dict:
        return {
            "high": float(self.high),
            "low": float(self.low),
            "open": float(self.open),
            "locked_at": self.locked_at.isoformat() if self.locked_at else None,
            "is_locked": self.is_locked,
            "width": float(self.width),
            "midpoint": float(self.midpoint),
            "bars_used": int(self.bars_used),
        }


def opening_range(
    df: pd.DataFrame,
    minutes: int = 30,
    session_start: Optional[time] = None,
) -> OpeningRange:
    """Compute the opening range from the most recent session in `df`.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with DatetimeIndex.
    minutes : int
        Length of the OR window in minutes. 30 is standard.
    session_start : datetime.time, optional
        Filter session to bars at or after this time. If None, uses the first
        bar of the most recent date in the index. Useful when bars include
        pre-market — pass `time(9, 30)` for US RTH or `time(9, 0)` for
        WDO regular session.

    Returns
    -------
    OpeningRange
        Locked once the window is past; still-forming if `df` ends mid-window.
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return OpeningRange()

    last_date = df.index[-1].date()
    session_mask = df.index.date == last_date
    session = df.loc[session_mask]

    if session_start is not None:
        session = session[session.index.time >= session_start]

    if session.empty:
        return OpeningRange()

    first_ts = session.index[0]
    or_end = first_ts + timedelta(minutes=minutes)
    or_window = session[session.index < or_end]

    if or_window.empty:
        return OpeningRange()

    open_price = float(or_window["open"].iloc[0])
    high = float(or_window["high"].max())
    low = float(or_window["low"].min())
    bars_used = len(or_window)

    # Locked iff there's a bar at or after or_end (which proves the window
    # has passed). Otherwise we're still inside it.
    has_post_window_bar = (session.index >= or_end).any()
    locked_at = or_end if has_post_window_bar else None

    return OpeningRange(
        high=high, low=low, open=open_price,
        locked_at=locked_at, bars_used=bars_used,
    )
