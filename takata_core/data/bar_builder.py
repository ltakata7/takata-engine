"""Tick-to-bar aggregation with time-based and tick-count modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import pandas as pd


@dataclass
class Bar:
    """Single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class Tick:
    """Single market tick."""
    timestamp: datetime
    price: float
    volume: float = 1.0


class BarBuilder:
    """Aggregates ticks into OHLCV bars.

    Parameters
    ----------
    mode : str
        ``"time"`` for time-based bars, ``"tick"`` for tick-count bars.
    interval : str | int
        For time mode: pandas frequency string (e.g. ``"5min"``).
        For tick mode: number of ticks per bar.
    on_bar : callable, optional
        Callback ``(Bar) -> None`` invoked when a bar completes.
    """

    def __init__(
        self,
        mode: str = "time",
        interval: str | int = "5min",
        on_bar: Callable[[Bar], None] | None = None,
    ):
        self.mode = mode
        self.interval = interval
        self.on_bar = on_bar

        self._ticks: list[Tick] = []
        self._bars: list[Bar] = []
        self._current_period_start: datetime | None = None

    def _get_period_start(self, ts: datetime) -> datetime:
        """Snap timestamp to its bar period start."""
        return pd.Timestamp(ts).floor(self.interval).to_pydatetime()

    def _flush_bar(self) -> Bar | None:
        """Build a bar from accumulated ticks and reset."""
        if not self._ticks:
            return None

        bar = Bar(
            timestamp=self._ticks[0].timestamp,
            open=self._ticks[0].price,
            high=max(t.price for t in self._ticks),
            low=min(t.price for t in self._ticks),
            close=self._ticks[-1].price,
            volume=sum(t.volume for t in self._ticks),
        )
        self._bars.append(bar)
        self._ticks.clear()

        if self.on_bar:
            self.on_bar(bar)
        return bar

    def process_tick(self, tick: Tick) -> Bar | None:
        """Ingest a tick, return a Bar if one completed.

        Parameters
        ----------
        tick : Tick
            Market tick to process.

        Returns
        -------
        Bar or None
            Completed bar, or None if the current bar is still building.
        """
        if self.mode == "time":
            period = self._get_period_start(tick.timestamp)
            if self._current_period_start is None:
                self._current_period_start = period

            if period > self._current_period_start:
                bar = self._flush_bar()
                self._current_period_start = period
                self._ticks.append(tick)
                return bar

            self._ticks.append(tick)
            return None

        elif self.mode == "tick":
            self._ticks.append(tick)
            if len(self._ticks) >= int(self.interval):
                return self._flush_bar()
            return None

        raise ValueError(f"Unknown mode: {self.mode}")

    def flush(self) -> Bar | None:
        """Force-emit the current partial bar (e.g. at session end)."""
        return self._flush_bar()

    def get_dataframe(self) -> pd.DataFrame:
        """Return all completed bars as a DataFrame."""
        if not self._bars:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame([b.to_dict() for b in self._bars])
        df = df.set_index("timestamp")
        return df

    @staticmethod
    def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample an OHLCV DataFrame to a coarser timeframe.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with DatetimeIndex.
        timeframe : str
            Target frequency (e.g. ``"15min"``, ``"1h"``).
        """
        return df.resample(timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open"])
