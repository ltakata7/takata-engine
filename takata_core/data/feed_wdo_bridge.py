"""WDO data feed via the Windows bridge HTTP API."""

from __future__ import annotations

import logging
import os
from typing import Iterator, Optional

import pandas as pd
import requests

from takata_core.data.base import DataFeed

logger = logging.getLogger(__name__)

DEFAULT_BRIDGE_URL = os.getenv("WDO_BRIDGE_URL", "http://127.0.0.1:8002")


class WDOBridgeFeed(DataFeed):
    """Fetches WDO OHLCV data from the Windows bridge server.

    Parameters
    ----------
    bridge_url : str
        URL of the wdo_bridge running on Windows.
    limit : int
        Number of bars to fetch.
    """

    def __init__(self, bridge_url: str = DEFAULT_BRIDGE_URL, limit: int = 200):
        self.bridge_url = bridge_url.rstrip("/")
        self.limit = limit
        self._df: Optional[pd.DataFrame] = None

    def is_available(self) -> bool:
        """Check if the WDO bridge is reachable."""
        try:
            r = requests.get(f"{self.bridge_url}/health", timeout=3)
            data = r.json()
            return data.get("connected", False)
        except Exception:
            return False

    def get_quote(self) -> dict:
        """Get the latest WDO quote."""
        r = requests.get(f"{self.bridge_url}/quote", timeout=5)
        return r.json()

    def load(self) -> pd.DataFrame:
        """Fetch OHLCV bars from the bridge."""
        if self._df is not None:
            return self._df

        r = requests.get(f"{self.bridge_url}/bars/ohlcv", params={"limit": self.limit}, timeout=10)
        data = r.json()

        if not data.get("timestamp"):
            raise ValueError("No bar data from WDO bridge. Is the market open?")

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        self._df = df
        logger.info("Loaded %d WDO bars from bridge", len(df))
        return self._df

    def replay(self) -> Iterator[pd.Series]:
        df = self.load()
        for ts, row in df.iterrows():
            yield row
