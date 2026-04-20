"""FRED Feed — Federal Reserve Economic Data.

API key required: https://fred.stlouisfed.org/docs/api/api_key.html
Set FRED_API_KEY in .env

Key series for MES trading and global macro analysis:
- GDP, Industrial Production → growth cycle
- CPI, PCE → inflation readings
- Unemployment, Payrolls, Claims → labor market
- Fed Funds, Treasury yields → monetary policy + term structure
- ISM PMI → manufacturing cycle
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from . import (
    EconomicIndicator,
    IndicatorBundle,
    MacroDataFeed,
    TTL_INTRADAY,
    TTL_DAILY,
    TTL_MONTHLY,
    TTL_QUARTERLY,
)

logger = logging.getLogger(__name__)

FRED_API = "https://api.stlouisfed.org/fred/series/observations"

# Series definitions: (series_id, name, category, frequency, unit, ttl)
FRED_SERIES = [
    # Growth
    ("GDPC1", "Real GDP", "growth", "quarterly", "Bil. 2017$", TTL_QUARTERLY),
    ("INDPRO", "Industrial Production Index", "growth", "monthly", "Index", TTL_MONTHLY),
    ("RSAFS", "Retail Sales", "growth", "monthly", "Mil. USD", TTL_MONTHLY),
    ("HOUST", "Housing Starts", "growth", "monthly", "Thousands", TTL_MONTHLY),
    ("UMCSENT", "Consumer Sentiment (UMich)", "growth", "monthly", "Index", TTL_MONTHLY),

    # Inflation
    ("CPIAUCSL", "CPI (All Items)", "inflation", "monthly", "Index", TTL_MONTHLY),
    ("CPILFESL", "Core CPI (ex Food/Energy)", "inflation", "monthly", "Index", TTL_MONTHLY),
    ("PCEPI", "PCE Price Index", "inflation", "monthly", "Index", TTL_MONTHLY),
    ("PCEPILFE", "Core PCE (ex Food/Energy)", "inflation", "monthly", "Index", TTL_MONTHLY),

    # Employment
    ("UNRATE", "Unemployment Rate", "employment", "monthly", "%", TTL_MONTHLY),
    ("ICSA", "Initial Jobless Claims", "employment", "weekly", "Thousands", TTL_DAILY),
    ("PAYEMS", "Nonfarm Payrolls", "employment", "monthly", "Thousands", TTL_MONTHLY),

    # Rates / Yields
    ("FEDFUNDS", "Fed Funds Rate", "rates", "monthly", "%", TTL_DAILY),
    ("DGS2", "2-Year Treasury Yield", "rates", "daily", "%", TTL_INTRADAY),
    ("DGS10", "10-Year Treasury Yield", "rates", "daily", "%", TTL_INTRADAY),
    ("DGS30", "30-Year Treasury Yield", "rates", "daily", "%", TTL_INTRADAY),
    ("T10Y2Y", "10Y-2Y Spread", "rates", "daily", "%", TTL_INTRADAY),
    ("T10Y3M", "10Y-3M Spread", "rates", "daily", "%", TTL_INTRADAY),

    # Activity
    # NOTE: NAPM (ISM PMI) was discontinued from FRED due to ISM licensing.
    # CFNAI is a 85-indicator composite — positive = above-trend growth, < -0.7 = recession onset.
    ("CFNAI", "Chicago Fed National Activity Index", "activity", "monthly", "Index", TTL_MONTHLY),
]


class FREDFeed(MacroDataFeed):
    """Fetches US economic indicators from FRED API."""

    source = "fred"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("FRED_API_KEY", "")

    def fetch_series(self, series_id: str, ttl: int = TTL_MONTHLY) -> Optional[Dict]:
        """Fetch a single series from FRED, with cache."""
        # Check cache first
        cached = self._read_cache(series_id, ttl)
        if cached and not cached.get("_stale"):
            return cached

        if not self.api_key:
            if cached:
                cached["_stale"] = True
                return cached
            return None

        try:
            # Fetch last 5 observations
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            }
            r = requests.get(FRED_API, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()

            observations = data.get("observations", [])
            if not observations:
                if cached:
                    cached["_stale"] = True
                    return cached
                return None

            # Parse observations (newest first due to sort_order=desc)
            parsed = []
            for obs in observations:
                try:
                    val = obs.get("value", ".")
                    if val == "." or val == "":
                        continue
                    parsed.append({
                        "date": obs["date"],
                        "value": float(val),
                    })
                except (ValueError, TypeError):
                    continue

            if not parsed:
                if cached:
                    cached["_stale"] = True
                    return cached
                return None

            # Reverse to chronological order (oldest first)
            parsed.reverse()

            result = {
                "series_id": series_id,
                "observations": parsed,
                "latest": parsed[-1],
                "previous": parsed[-2] if len(parsed) >= 2 else parsed[-1],
            }

            self._write_cache(series_id, result)
            return result

        except Exception as e:
            logger.warning("FRED fetch failed for %s: %s", series_id, e)
            if cached:
                cached["_stale"] = True
                return cached
            return None

    def fetch_all(self) -> IndicatorBundle:
        """Fetch all FRED series and return as an IndicatorBundle."""
        bundle = IndicatorBundle(source="fred")

        if not self.api_key:
            bundle.errors.append("FRED_API_KEY not configured. Set it in .env to enable US macro data.")
            logger.warning("FRED API key not set — skipping US macro data")
            return bundle

        errors = []

        for series_id, name, category, frequency, unit, ttl in FRED_SERIES:
            data = self.fetch_series(series_id, ttl)

            if data is None:
                errors.append(f"Failed to fetch {name} ({series_id})")
                continue

            latest = data["latest"]
            previous = data["previous"]
            value = latest["value"]
            prev_value = previous["value"]

            # For index series (CPI, PCE, INDPRO), compute % change
            if unit == "Index":
                change = ((value / prev_value) - 1) * 100 if prev_value != 0 else 0
                change_pct = change
            else:
                change = value - prev_value
                change_pct = (change / abs(prev_value) * 100) if prev_value != 0 else 0

            signal, interpretation = self._interpret_change(name, category, change, value)

            stale = data.get("_stale", False)
            if stale:
                bundle.stale = True

            indicator = EconomicIndicator(
                series_id=series_id,
                name=name,
                source="fred",
                category=category,
                latest_value=value,
                previous_value=prev_value,
                change=round(change, 4),
                change_pct=round(change_pct, 4),
                date=latest["date"],
                frequency=frequency,
                unit=unit,
                signal=signal,
                interpretation=interpretation,
                ttl=ttl,
            )
            bundle.indicators.append(indicator)

            # Rate limit: 0.3s between FRED requests
            time.sleep(0.3)

        bundle.errors = errors
        if bundle.indicators:
            logger.info("FRED feed: loaded %d indicators (%d errors)", len(bundle.indicators), len(errors))
        return bundle
