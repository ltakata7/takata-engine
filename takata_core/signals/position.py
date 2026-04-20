"""Position tracking for open and closed trades."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Position:
    """Represents a single trade (open or closed)."""

    instrument: str
    side: str  # "long" or "short"
    entry_price: float
    entry_time: datetime
    size: float
    stop: float
    target: float
    regime: str
    status: str = "open"
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    def close(self, price: float, time: datetime, reason: str) -> None:
        """Close the position and compute P&L."""
        self.exit_price = price
        self.exit_time = time
        self.exit_reason = reason
        self.status = "closed"

        if self.side == "long":
            self.pnl = (price - self.entry_price) * self.size
        else:
            self.pnl = (self.entry_price - price) * self.size

    def unrealized_pnl(self, current_price: float) -> float:
        """Compute unrealized P&L at current price."""
        if not self.is_open:
            return self.pnl
        if self.side == "long":
            return (current_price - self.entry_price) * self.size
        return (self.entry_price - current_price) * self.size

    def check_stop(self, bar: pd.Series) -> bool:
        """Check if stop was hit during this bar."""
        if self.side == "long":
            return bar["low"] <= self.stop
        return bar["high"] >= self.stop

    def check_target(self, bar: pd.Series) -> bool:
        """Check if target was hit during this bar."""
        if self.side == "long":
            return bar["high"] >= self.target
        return bar["low"] <= self.target


class PositionManager:
    """Manages open and closed positions for an instrument.

    Parameters
    ----------
    point_value : float
        Dollar/BRL value per point (e.g. 5.0 for MES, 10.0 for WDO).
    """

    def __init__(self, point_value: float = 1.0):
        self.point_value = point_value
        self._open: Dict[str, Position] = {}
        self._closed: List[Position] = []

    def has_open(self, instrument: str) -> bool:
        return instrument in self._open

    def get_open(self, instrument: str) -> Optional[Position]:
        return self._open.get(instrument)

    def open_position(
        self,
        instrument: str,
        side: str,
        entry_price: float,
        entry_time: datetime,
        size: float,
        stop: float,
        target: float,
        regime: str,
    ) -> Position:
        """Open a new position. Closes any existing position for the instrument first."""
        if instrument in self._open:
            # Force close existing position at current price
            existing = self._open[instrument]
            existing.close(entry_price, entry_time, "replaced")
            self._closed.append(existing)

        pos = Position(
            instrument=instrument,
            side=side,
            entry_price=entry_price,
            entry_time=entry_time,
            size=size,
            stop=stop,
            target=target,
            regime=regime,
        )
        self._open[instrument] = pos
        return pos

    def check_exits(self, instrument: str, bar: pd.Series, bar_time: datetime) -> Optional[Position]:
        """Check if stop or target was hit. Returns closed Position or None.

        Stop is checked first (conservative — assume worst fill in same bar).
        """
        pos = self._open.get(instrument)
        if pos is None:
            return None

        if pos.check_stop(bar):
            pos.close(pos.stop, bar_time, "stop_hit")
            pos.pnl *= self.point_value
            del self._open[instrument]
            self._closed.append(pos)
            return pos

        if pos.check_target(bar):
            pos.close(pos.target, bar_time, "target_hit")
            pos.pnl *= self.point_value
            del self._open[instrument]
            self._closed.append(pos)
            return pos

        return None

    def close_position(
        self, instrument: str, price: float, time: datetime, reason: str
    ) -> Optional[Position]:
        """Manually close a position."""
        pos = self._open.get(instrument)
        if pos is None:
            return None

        pos.close(price, time, reason)
        pos.pnl *= self.point_value
        del self._open[instrument]
        self._closed.append(pos)
        return pos

    def daily_pnl(self, date: Optional[datetime] = None) -> float:
        """Sum realized P&L for closed positions on a given date (or all)."""
        if date is None:
            return sum(p.pnl for p in self._closed)
        target_date = date.date() if hasattr(date, "date") else date
        return sum(
            p.pnl for p in self._closed
            if p.exit_time and p.exit_time.date() == target_date
        )

    @property
    def closed_positions(self) -> List[Position]:
        return list(self._closed)

    @property
    def trade_count(self) -> int:
        return len(self._closed)
