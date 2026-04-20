"""CSV data feed for backtesting with auto-format detection."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from takata_engine.data.base import DataFeed

# Common column name mappings from various data sources
_COLUMN_ALIASES = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "Date": "timestamp",
    "DateTime": "timestamp",
    "Timestamp": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Vol": "volume",
    "vol": "volume",
    "Adj Close": "adj_close",
}

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


class CSVFeed(DataFeed):
    """Loads OHLCV bar data from a CSV file.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    sep : str
        Column separator. Default ``,``.
    parse_dates : bool
        Whether to parse the timestamp column. Default ``True``.
    tz : str | None
        Timezone to localize naive timestamps (e.g. ``"America/New_York"``).
    """

    def __init__(
        self,
        path: str | Path,
        sep: str = ",",
        parse_dates: bool = True,
        tz: str | None = None,
    ):
        self.path = Path(path)
        self.sep = sep
        self.parse_dates = parse_dates
        self.tz = tz
        self._df: pd.DataFrame | None = None

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to standard OHLCV names."""
        df = df.rename(columns=_COLUMN_ALIASES)
        df.columns = df.columns.str.lower().str.strip()
        # Second pass after lowering
        df = df.rename(columns={c: _COLUMN_ALIASES.get(c, c) for c in df.columns})
        return df

    def load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        df = pd.read_csv(self.path, sep=self.sep)
        df = self._normalize_columns(df)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        # Set up timestamp index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()
        elif self.parse_dates and df.index.dtype == "object":
            df.index = pd.to_datetime(df.index)
            df.index.name = "timestamp"
            df = df.sort_index()

        if self.tz and df.index.tz is None:
            df.index = df.index.tz_localize(self.tz)

        # Ensure numeric types
        for col in REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        self._df = df
        return self._df

    def replay(self) -> Iterator[pd.Series]:
        df = self.load()
        for ts, row in df.iterrows():
            yield row
