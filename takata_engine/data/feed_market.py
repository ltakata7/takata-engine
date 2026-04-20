"""Universal market data feed via yfinance for stocks, ETFs, and REITs."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import yfinance as yf

from takata_engine.data.base import DataFeed

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache"


class MarketDataFeed(DataFeed):
    """Fetches OHLCV data for any US ticker via yfinance.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. ``"AAPL"``, ``"SPY"``, ``"O"``).
    period : str
        Data period (``"1y"``, ``"2y"``, ``"5y"``, ``"max"``).
    interval : str
        Bar interval (``"1d"``, ``"1wk"``, ``"1mo"``).
    cache : bool
        Whether to cache data locally.
    """

    def __init__(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        cache: bool = True,
    ):
        self.ticker = ticker.upper()
        self.period = period
        self.interval = interval
        self.use_cache = cache
        self._df: Optional[pd.DataFrame] = None

    def _cache_path(self) -> Path:
        key = f"{self.ticker}_{self.period}_{self.interval}"
        h = hashlib.md5(key.encode()).hexdigest()[:8]
        return CACHE_DIR / f"{self.ticker}_{self.interval}_{h}.csv"

    def load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        cache_path = self._cache_path()

        # Check cache (valid for 1 day for daily data)
        if self.use_cache and cache_path.exists():
            import time
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            max_age = 24 if self.interval == "1d" else 168  # 1 day or 1 week
            if age_hours < max_age:
                logger.info("Loading %s from cache", self.ticker)
                self._df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                self._df.index.name = "timestamp"
                return self._df

        logger.info("Fetching %s (%s, %s) from yfinance", self.ticker, self.period, self.interval)
        t = yf.Ticker(self.ticker)
        df = t.history(period=self.period, interval=self.interval)

        if df.empty:
            raise ValueError(f"No data returned for ticker '{self.ticker}'")

        # Normalize columns to lowercase
        df.columns = df.columns.str.lower()
        df.index.name = "timestamp"

        # Keep only OHLCV
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]

        # Cache
        if self.use_cache:
            CACHE_DIR.mkdir(exist_ok=True)
            df.to_csv(cache_path)

        self._df = df
        return self._df

    def replay(self) -> Iterator[pd.Series]:
        df = self.load()
        for ts, row in df.iterrows():
            yield row

    def load_multi_timeframe(self) -> dict:
        """Load daily, weekly, and monthly data for the same ticker.

        Returns
        -------
        dict
            Keys: ``"daily"``, ``"weekly"``, ``"monthly"`` → DataFrames.
        """
        result = {}
        for interval, period in [("1d", self.period), ("1wk", "2y"), ("1mo", "5y")]:
            feed = MarketDataFeed(self.ticker, period=period, interval=interval, cache=self.use_cache)
            try:
                result[{"1d": "daily", "1wk": "weekly", "1mo": "monthly"}[interval]] = feed.load()
            except ValueError:
                logger.warning("Could not load %s data for %s", interval, self.ticker)
        return result
