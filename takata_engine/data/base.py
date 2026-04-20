"""Abstract base class for data feeds."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import pandas as pd


class DataFeed(ABC):
    """Base interface for all data feeds (CSV, IBKR, Profit RTD)."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the full dataset as a DataFrame.

        Returns
        -------
        pd.DataFrame
            OHLCV data with a DatetimeIndex named ``timestamp`` and columns:
            ``open``, ``high``, ``low``, ``close``, ``volume``.
        """

    @abstractmethod
    def replay(self) -> Iterator[pd.Series]:
        """Yield bars one at a time for event-driven backtesting.

        Yields
        ------
        pd.Series
            Single bar with index ``[open, high, low, close, volume]``
            and ``.name`` set to the bar timestamp.
        """
