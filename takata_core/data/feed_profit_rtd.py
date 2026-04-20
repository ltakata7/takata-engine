"""Profit Ultra RTD/COM data feed for WDO (B3) — Windows only.

Uses pywin32 COM interface to connect to Nelogica Profit Ultra's
RTD server for real-time tick data on Mini Dólar (WDOFUT) and
other B3 instruments.

NOTE: This module only works on Windows with Profit Ultra installed.
On Mac/Linux it provides a stub that raises NotImplementedError.
"""

from __future__ import annotations

import logging
import platform
from typing import Callable, Iterator, Optional

import pandas as pd

from takata_core.data.base import DataFeed

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"


class ProfitRTDFeed(DataFeed):
    """Connects to Profit Ultra via COM/RTD for real-time WDO data.

    Parameters
    ----------
    instrument : str
        B3 instrument code (e.g. ``"WDOFUT"``, ``"WINFUT"``).
    server : str
        RTD server ProgID.
    """

    def __init__(
        self,
        instrument: str = "WDOFUT",
        server: str = "RTDProfitChartServer",
    ):
        self.instrument = instrument.upper()
        self.server = server
        self._connected = False

    def connect(self) -> None:
        """Establish RTD connection to Profit Ultra."""
        if not IS_WINDOWS:
            raise NotImplementedError(
                "Profit Ultra RTD requires Windows. "
                "Run this on your Windows desktop with Profit Ultra installed."
            )

        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        self._rtd = win32com.client.Dispatch(self.server)
        self._connected = True
        logger.info("Connected to Profit Ultra RTD: %s", self.server)

    def load(self) -> pd.DataFrame:
        """Load historical intraday data from Profit Ultra.

        For backtesting on Windows, Profit Ultra can export CSV data.
        This method is a placeholder — use CSVFeed with Profit exports.
        """
        raise NotImplementedError(
            "ProfitRTDFeed.load() not available for bulk historical data. "
            "Export CSV from Profit Ultra and use CSVFeed instead."
        )

    def replay(self) -> Iterator[pd.Series]:
        raise NotImplementedError("Use CSVFeed for replay mode.")

    def get_quote(self, field: str = "ULT") -> Optional[float]:
        """Get a single real-time quote field.

        Parameters
        ----------
        field : str
            RTD field: ``"ULT"`` (last), ``"MAX"`` (high), ``"MIN"`` (low),
            ``"ABE"`` (open), ``"VOL"`` (volume), ``"MEL_C"`` (best bid),
            ``"MEL_V"`` (best ask).
        """
        if not self._connected:
            self.connect()

        try:
            value = self._rtd.Cells(self.instrument, field)
            return float(value) if value is not None else None
        except Exception as e:
            logger.error("RTD quote error for %s.%s: %s", self.instrument, field, e)
            return None

    def stream_ticks(
        self,
        on_tick: Callable[[dict], None],
        interval_ms: int = 500,
    ) -> None:
        """Stream real-time ticks from Profit Ultra.

        Parameters
        ----------
        on_tick : callable
            Called with dict: ``{timestamp, price, volume, bid, ask}``.
        interval_ms : int
            Polling interval in milliseconds.
        """
        if not IS_WINDOWS:
            raise NotImplementedError("Profit Ultra RTD requires Windows.")

        import time
        from datetime import datetime

        if not self._connected:
            self.connect()

        logger.info("Streaming %s ticks (interval=%dms)...", self.instrument, interval_ms)

        try:
            while True:
                tick = {
                    "timestamp": datetime.now(),
                    "price": self.get_quote("ULT"),
                    "volume": self.get_quote("VOL"),
                    "bid": self.get_quote("MEL_C"),
                    "ask": self.get_quote("MEL_V"),
                    "high": self.get_quote("MAX"),
                    "low": self.get_quote("MIN"),
                }
                on_tick(tick)
                time.sleep(interval_ms / 1000)
        except KeyboardInterrupt:
            logger.info("Tick stream stopped")
