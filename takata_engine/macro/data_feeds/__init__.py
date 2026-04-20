"""Macro economic data feeds — shared types and cache layer.

Provides base infrastructure for FRED, BCB, and yield curve data sources.
All feeds use file-based JSON caching with configurable TTLs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "cache" / "macro"

# TTLs by data frequency (seconds)
TTL_INTRADAY = 6 * 3600      # 6h — daily yields, rates
TTL_DAILY = 12 * 3600         # 12h — daily indicators
TTL_MONTHLY = 24 * 3600       # 24h — CPI, IPCA, unemployment
TTL_QUARTERLY = 48 * 3600     # 48h — GDP


@dataclass
class EconomicIndicator:
    """A single economic data point with interpretation."""
    series_id: str
    name: str
    source: str              # "fred", "bcb", "yield_curve"
    category: str            # "growth", "inflation", "employment", "rates", "activity", "fx"
    latest_value: float
    previous_value: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0
    date: str = ""           # date of latest observation
    frequency: str = ""      # "daily", "monthly", "quarterly"
    unit: str = ""           # "%", "index", "USD billions", etc.
    signal: str = "neutral"  # "bullish", "bearish", "neutral"
    interpretation: str = ""
    ttl: int = TTL_MONTHLY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_id": self.series_id,
            "name": self.name,
            "source": self.source,
            "category": self.category,
            "latest_value": self.latest_value,
            "previous_value": self.previous_value,
            "change": round(self.change, 4),
            "change_pct": round(self.change_pct, 4),
            "date": self.date,
            "frequency": self.frequency,
            "unit": self.unit,
            "signal": self.signal,
            "interpretation": self.interpretation,
        }


@dataclass
class IndicatorBundle:
    """Collection of economic indicators from one source."""
    source: str
    indicators: List[EconomicIndicator] = field(default_factory=list)
    fetched_at: str = ""
    errors: List[str] = field(default_factory=list)
    stale: bool = False

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.now().isoformat()

    def get(self, series_id: str) -> Optional[EconomicIndicator]:
        """Get indicator by series ID."""
        for ind in self.indicators:
            if ind.series_id == series_id:
                return ind
        return None

    def by_category(self, category: str) -> List[EconomicIndicator]:
        """Get all indicators in a category."""
        return [i for i in self.indicators if i.category == category]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "count": len(self.indicators),
            "fetched_at": self.fetched_at,
            "stale": self.stale,
            "errors": self.errors,
            "categories": list(set(i.category for i in self.indicators)),
            "indicators": [i.to_dict() for i in self.indicators],
        }


class MacroDataFeed:
    """Base class for economic data feeds with JSON file caching."""

    source: str = "base"

    def __init__(self):
        self._cache_dir = CACHE_DIR / self.source
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, series_id: str) -> Path:
        safe_id = series_id.replace("/", "_").replace("\\", "_")
        return self._cache_dir / f"{safe_id}.json"

    def _read_cache(self, series_id: str, ttl: int) -> Optional[Dict]:
        """Read from cache if valid."""
        path = self._cache_path(series_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            cached_at = data.get("_cached_at", 0)
            if time.time() - cached_at < ttl:
                return data
            # Stale but exists — return with stale flag for fallback
            data["_stale"] = True
            return data
        except (json.JSONDecodeError, KeyError):
            return None

    def _write_cache(self, series_id: str, data: Dict) -> None:
        """Write data to cache."""
        data["_cached_at"] = time.time()
        data["_stale"] = False
        try:
            self._cache_path(series_id).write_text(json.dumps(data, default=str))
        except Exception as e:
            logger.warning("Cache write failed for %s/%s: %s", self.source, series_id, e)

    def _interpret_change(self, name: str, category: str, change: float, value: float) -> tuple:
        """Generate signal and interpretation from indicator change.

        Returns (signal, interpretation) tuple.
        """
        if abs(change) < 0.01:
            return "neutral", f"{name} unchanged"

        direction = "rising" if change > 0 else "falling"

        # Category-specific interpretation
        if category == "growth":
            signal = "bullish" if change > 0 else "bearish"
            interp = f"{name} {direction} ({change:+.2f}) — {'expanding' if change > 0 else 'contracting'} economy"
        elif category == "inflation":
            # Rising inflation = bearish (hawkish policy), falling = bullish
            signal = "bearish" if change > 0 else "bullish"
            interp = f"{name} {direction} ({change:+.2f}%) — {'inflationary pressure' if change > 0 else 'disinflation'}"
        elif category == "employment":
            if "unemployment" in name.lower() or "claims" in name.lower():
                signal = "bearish" if change > 0 else "bullish"
                interp = f"{name} {direction} — {'weakening' if change > 0 else 'strengthening'} labor market"
            else:
                signal = "bullish" if change > 0 else "bearish"
                interp = f"{name} {direction} — {'job growth' if change > 0 else 'job losses'}"
        elif category == "rates":
            signal = "neutral"
            interp = f"{name} at {value:.2f}% ({direction})"
        elif category == "activity":
            # CFNAI: 0 = trend growth, >0 = above-trend, <-0.7 = recession onset
            # PMI-style: 50 = neutral (legacy, kept for future if ISM returns)
            if "cfnai" in name.lower() or abs(value) < 5:
                signal = "bullish" if value > 0 else "bearish" if value < -0.3 else "neutral"
                interp = f"{name} at {value:.2f} — {'above trend' if value > 0 else 'below trend' if value < 0 else 'at trend'}"
            else:
                signal = "bullish" if value > 50 else "bearish" if value < 50 else "neutral"
                interp = f"{name} at {value:.1f} — {'expansion' if value > 50 else 'contraction'}"
        else:
            signal = "neutral"
            interp = f"{name} {direction}"

        return signal, interp
