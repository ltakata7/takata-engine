"""BCB SGS Feed — Brazilian Central Bank economic data.

Free API, no key required: https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados
Returns: [{data: "DD/MM/YYYY", valor: "123.45"}, ...]

Key series for WDO trading and macro analysis:
- Selic, CDI → monetary policy stance
- IPCA, IGP-M → inflation readings
- GDP, Unemployment → growth cycle
- Trade Balance, Reserves → external accounts
- Focus surveys → market consensus expectations
- EMBI+ → country risk premium
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from . import (
    EconomicIndicator,
    IndicatorBundle,
    MacroDataFeed,
    TTL_DAILY,
    TTL_MONTHLY,
    TTL_QUARTERLY,
)

logger = logging.getLogger(__name__)

BCB_API = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"
BCB_LAST_N = "&dataInicial={start}"  # DD/MM/YYYY

# Series definitions: (code, name, category, frequency, unit, ttl)
BCB_SERIES = [
    # Monetary Policy
    (432, "Selic Target Rate", "rates", "daily", "%", TTL_DAILY),
    (12, "CDI Rate", "rates", "daily", "% p.a.", TTL_DAILY),

    # Inflation
    (433, "IPCA (CPI Brazil)", "inflation", "monthly", "% m/m", TTL_MONTHLY),
    (7478, "IPCA-15 (Preview CPI)", "inflation", "monthly", "% m/m", TTL_MONTHLY),
    (189, "IGP-M (Wholesale Prices)", "inflation", "monthly", "% m/m", TTL_MONTHLY),

    # Growth
    # NOTE: 4380 is GDP LEVEL in R$ millions (not a growth rate). Growth % is
    # computed from q/q change. The legacy "growth rate" codes (7326, 22099,
    # 22109) all return 404 — use Olinda OData in a later phase for y/y rates.
    (4380, "GDP Level (quarterly)", "growth", "quarterly", "R$ Mil", TTL_QUARTERLY),
    (24369, "Unemployment Rate", "employment", "monthly", "%", TTL_MONTHLY),

    # External
    (22707, "Trade Balance", "fx", "monthly", "USD millions", TTL_MONTHLY),
    (3546, "Foreign Reserves", "fx", "monthly", "USD millions", TTL_MONTHLY),

    # Risk
    (40940, "EMBI+ Brazil", "rates", "daily", "bps", TTL_DAILY),

    # Focus Survey (Market Expectations)
    # NOTE: Only 13521 still works via legacy SGS. Focus Selic (13522) and Focus GDP
    # (13502) moved to the Olinda OData API — add as separate feed in a later phase.
    (13521, "Focus: IPCA Expectation (12m)", "inflation", "weekly", "%", TTL_DAILY),
]


class BCBFeed(MacroDataFeed):
    """Fetches Brazilian economic indicators from BCB SGS API."""

    source = "bcb"

    def fetch_series(self, code: int, ttl: int = TTL_MONTHLY) -> Optional[Dict]:
        """Fetch a single series from BCB, with cache."""
        series_id = str(code)

        # Check cache first
        cached = self._read_cache(series_id, ttl)
        if cached and not cached.get("_stale"):
            return cached

        # Fetch from API — last 2 observations for change calculation
        try:
            # Get last 90 days of data to ensure at least 2 observations
            from datetime import timedelta
            start = (datetime.now() - timedelta(days=90)).strftime("%d/%m/%Y")
            url = BCB_API.format(code=code) + f"&dataInicial={start}"

            r = requests.get(url, timeout=10, headers={"Accept": "application/json"})
            r.raise_for_status()
            data = r.json()

            if not data or not isinstance(data, list):
                # Return stale cache if available
                if cached:
                    cached["_stale"] = True
                    return cached
                return None

            # Parse: BCB returns [{data: "DD/MM/YYYY", valor: "123.45"}, ...]
            observations = []
            for obs in data:
                try:
                    date_str = obs.get("data", "")
                    val_str = obs.get("valor", "")
                    if val_str and val_str != "-":
                        val = float(val_str.replace(",", "."))
                        # Convert DD/MM/YYYY to YYYY-MM-DD
                        parts = date_str.split("/")
                        if len(parts) == 3:
                            iso_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
                        else:
                            iso_date = date_str
                        observations.append({"date": iso_date, "value": val})
                except (ValueError, TypeError):
                    continue

            if not observations:
                if cached:
                    cached["_stale"] = True
                    return cached
                return None

            result = {
                "code": code,
                "observations": observations[-10:],  # keep last 10
                "latest": observations[-1],
                "previous": observations[-2] if len(observations) >= 2 else observations[-1],
            }

            self._write_cache(series_id, result)
            return result

        except Exception as e:
            logger.warning("BCB fetch failed for series %d: %s", code, e)
            if cached:
                cached["_stale"] = True
                return cached
            return None

    def fetch_all(self) -> IndicatorBundle:
        """Fetch all BCB series and return as an IndicatorBundle."""
        bundle = IndicatorBundle(source="bcb")
        errors = []

        for code, name, category, frequency, unit, ttl in BCB_SERIES:
            data = self.fetch_series(code, ttl)

            if data is None:
                errors.append(f"Failed to fetch {name} (SGS {code})")
                continue

            latest = data["latest"]
            previous = data["previous"]
            value = latest["value"]
            prev_value = previous["value"]
            change = value - prev_value
            change_pct = (change / abs(prev_value) * 100) if prev_value != 0 else 0

            signal, interpretation = self._interpret_change(name, category, change, value)

            # Stale flag
            stale = data.get("_stale", False)
            if stale:
                bundle.stale = True

            indicator = EconomicIndicator(
                series_id=f"BCB_{code}",
                name=name,
                source="bcb",
                category=category,
                latest_value=value,
                previous_value=prev_value,
                change=change,
                change_pct=change_pct,
                date=latest["date"],
                frequency=frequency,
                unit=unit,
                signal=signal,
                interpretation=interpretation,
                ttl=ttl,
            )
            bundle.indicators.append(indicator)

            # Rate limit: 0.5s between requests
            time.sleep(0.5)

        bundle.errors = errors
        if bundle.indicators:
            logger.info("BCB feed: loaded %d indicators (%d errors)", len(bundle.indicators), len(errors))
        return bundle
