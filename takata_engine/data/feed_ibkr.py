"""IBKR live data feed via ib_insync — real-time and historical bars for MES/ES/NQ."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional

import pandas as pd
from ib_insync import IB, Contract, Future, Stock, util

from takata_engine.data.bar_builder import Bar, BarBuilder, Tick
from takata_engine.data.base import DataFeed

logger = logging.getLogger(__name__)

# Default connection settings
import os

DEFAULT_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("IBKR_PORT", "4001"))
DEFAULT_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "10"))


def _front_month_future(symbol: str, exchange: str = "CME") -> Future:
    """Create a front-month futures contract.

    Uses an empty expiry — ib_insync will resolve to the front month
    when qualified via the connection. If ambiguous, picks the nearest.
    """
    return Future(symbol, exchange=exchange, currency="USD")


# Instrument → contract mapping
CONTRACT_MAP = {
    "MES": lambda: _front_month_future("MES", "CME"),
    "ES": lambda: _front_month_future("ES", "CME"),
    "NQ": lambda: _front_month_future("NQ", "CME"),
    "MNQ": lambda: _front_month_future("MNQ", "CME"),
    "MYM": lambda: _front_month_future("MYM", "CBOT"),
}


class IBKRConnection:
    """Manages a singleton IBKR connection.

    Parameters
    ----------
    host : str
        TWS/Gateway hostname.
    port : int
        API port (4001=Gateway live, 4002=Gateway paper, 7497=TWS paper).
    client_id : int
        Unique client ID for this connection.
    """

    _instance: Optional["IBKRConnection"] = None

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, client_id: int = DEFAULT_CLIENT_ID):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()

    @classmethod
    def get(cls, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, client_id: int = DEFAULT_CLIENT_ID) -> "IBKRConnection":
        if cls._instance is None or not cls._instance.ib.isConnected():
            cls._instance = cls(host, port, client_id)
            cls._instance.connect()
        return cls._instance

    def connect(self) -> None:
        if self.ib.isConnected():
            return
        logger.info("Connecting to IBKR at %s:%d (client_id=%d)", self.host, self.port, self.client_id)
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        logger.info("IBKR connected: %s", self.ib.managedAccounts())

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("IBKR disconnected")

    @property
    def connected(self) -> bool:
        return self.ib.isConnected()


def get_contract(instrument: str, ib: Optional[IB] = None) -> Contract:
    """Create an IBKR contract for the given instrument.

    For futures, resolves to front-month by querying contract details.
    """
    instrument = instrument.upper()
    factory = CONTRACT_MAP.get(instrument)
    if factory is None:
        contract = Stock(instrument, "SMART", "USD")
        if ib and ib.isConnected():
            ib.qualifyContracts(contract)
        return contract

    contract = factory()

    # For futures, always resolve via reqContractDetails to get front month
    if ib and ib.isConnected():
        details = ib.reqContractDetails(contract)
        if details:
            # Sort by expiry, pick the nearest (front month)
            details.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
            contract = details[0].contract
            ib.qualifyContracts(contract)
            logger.info("Resolved %s → %s (expiry %s)",
                        instrument, contract.localSymbol, contract.lastTradeDateOrContractMonth)

    return contract


class IBKRDataFeed(DataFeed):
    """Fetches historical and live OHLCV data from Interactive Brokers.

    Parameters
    ----------
    instrument : str
        Instrument symbol (e.g. ``"MES"``, ``"ES"``).
    host : str
        TWS/Gateway hostname.
    port : int
        API port.
    duration : str
        Historical data duration (e.g. ``"1 D"``, ``"5 D"``, ``"1 M"``).
    bar_size : str
        Bar size (e.g. ``"5 mins"``, ``"1 min"``, ``"15 mins"``, ``"1 day"``).
    """

    def __init__(
        self,
        instrument: str = "MES",
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        duration: str = "1 D",
        bar_size: str = "5 mins",
    ):
        self.instrument = instrument.upper()
        self.host = host
        self.port = port
        self.duration = duration
        self.bar_size = bar_size
        self._df: Optional[pd.DataFrame] = None

    def _get_connection(self) -> IBKRConnection:
        return IBKRConnection.get(self.host, self.port)

    def load(self) -> pd.DataFrame:
        """Fetch historical bars from IBKR."""
        if self._df is not None:
            return self._df

        conn = self._get_connection()
        contract = get_contract(self.instrument, conn.ib)
        logger.info("Fetching %s historical data: %s / %s", self.instrument, self.duration, self.bar_size)

        bars = conn.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=self.duration,
            barSizeSetting=self.bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

        if not bars:
            raise ValueError(f"No historical data returned for {self.instrument}")

        df = util.df(bars)
        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Standardize columns
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        self._df = df[keep]
        logger.info("Loaded %d bars for %s", len(self._df), self.instrument)
        return self._df

    def replay(self) -> Iterator[pd.Series]:
        df = self.load()
        for ts, row in df.iterrows():
            yield row

    def stream_bars(
        self,
        on_bar: Callable[[pd.Series], None],
        bar_size: str = "5 mins",
    ) -> None:
        """Stream real-time bars from IBKR.

        Parameters
        ----------
        on_bar : callable
            Callback invoked with each new bar (pd.Series).
        bar_size : str
            Real-time bar size.
        """
        conn = self._get_connection()
        contract = get_contract(self.instrument)
        conn.ib.qualifyContracts(contract)

        logger.info("Starting real-time bar stream for %s", self.instrument)

        bars = conn.ib.reqRealTimeBars(contract, barSize=5, whatToShow="TRADES", useRTH=True)

        def on_bar_update(bars, hasNewBar):
            if hasNewBar:
                bar = bars[-1]
                series = pd.Series({
                    "open": bar.open_,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }, name=pd.Timestamp(bar.time))
                on_bar(series)

        bars.updateEvent += on_bar_update

        # Run the event loop
        logger.info("Streaming... Press Ctrl+C to stop")
        try:
            conn.ib.run()
        except KeyboardInterrupt:
            conn.ib.cancelRealTimeBars(bars)
            logger.info("Stream stopped")
