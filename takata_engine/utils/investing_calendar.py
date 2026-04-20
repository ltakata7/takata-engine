"""Investing.com Economic Calendar scraper — live event feed.

Fetches upcoming economic events from investing.com's AJAX calendar
endpoint and converts them to ``EconomicEvent`` objects that the existing
``CalendarFilter`` can consume for signal gating.

**Why live calendar matters for WDO/MES 3-5min trading:**
- Hardcoded event dates go stale after 1 week
- New events get added (emergency Fed speeches, BCB interventions)
- Actual/Forecast/Previous values inform post-release signal interpretation
- CFTC positioning data (Friday) affects Monday's opening flow

The scraper:
1. Fetches next 7 days of USD + BRL events from investing.com
2. Caches result to disk (4h TTL — events update intraday)
3. Converts to EconomicEvent objects for CalendarFilter
4. Provides a list endpoint for frontend rendering

Usage::

    from takata_engine.utils.investing_calendar import get_calendar_events, refresh_calendar

    # Auto-fetch (cached)
    events = get_calendar_events()

    # Force refresh
    events = refresh_calendar()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Config ──
INVESTING_URL = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "calendar"
CACHE_FILE = CACHE_DIR / "investing_calendar.json"
CACHE_TTL = 4 * 3600  # 4 hours

# Currencies we care about
RELEVANT_CURRENCIES = {"USD", "BRL"}

# Impact mapping: investing.com uses 1-3 "bull" icons
IMPACT_MAP = {
    3: "high",
    2: "medium",
    1: "low",
    0: "low",
}

# Blackout window defaults by impact level (minutes before/after)
BLACKOUT_DEFAULTS = {
    "high": (20, 20),
    "medium": (10, 10),
    "low": (5, 5),
}

# Events that deserve extra-wide blackout windows
WIDE_BLACKOUT_KEYWORDS = {
    "fomc": (30, 60),
    "nfp": (30, 30),
    "nonfarm": (30, 30),
    "copom": (60, 60),
    "selic": (60, 60),
    "cpi": (15, 15),
    "ipca": (15, 15),
    "gdp": (15, 15),
    "pce": (15, 15),
    "trump speaks": (15, 30),
    "fed chair": (15, 30),
    "powell": (15, 30),
}

# Headers to look like a real browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*",
    "Referer": "https://www.investing.com/economic-calendar/",
}


@dataclass
class CalendarEvent:
    """Parsed economic calendar event from investing.com."""
    event_id: str
    name: str
    datetime_utc: str       # "2026-04-16T12:30:00"
    currency: str           # "USD", "BRL"
    impact: str             # "high", "medium", "low"
    impact_stars: int       # 1-3
    actual: str = ""
    forecast: str = ""
    previous: str = ""
    # Derived
    affects: List[str] = field(default_factory=list)
    blackout_before_min: int = 10
    blackout_after_min: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "name": self.name,
            "datetime_utc": self.datetime_utc,
            "currency": self.currency,
            "impact": self.impact,
            "impact_stars": self.impact_stars,
            "actual": self.actual,
            "forecast": self.forecast,
            "previous": self.previous,
            "affects": self.affects,
            "blackout_before_min": self.blackout_before_min,
            "blackout_after_min": self.blackout_after_min,
        }


def _fetch_from_investing(days_ahead: int = 7) -> List[CalendarEvent]:
    """Fetch economic calendar from investing.com AJAX endpoint.

    Returns parsed list of CalendarEvent objects for USD + BRL.
    """
    today = datetime.utcnow()
    form_data = {
        "dateFrom": today.strftime("%Y-%m-%d"),
        "dateTo": (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d"),
        "timeZone": 55,  # UTC
        "timeFilter": "timeRemain",
        "currentTab": "custom",
        "limit_from": 0,
    }

    try:
        r = requests.post(INVESTING_URL, headers=HEADERS, data=form_data, timeout=15)
        r.raise_for_status()
        result = r.json()
    except Exception as e:
        logger.warning("Investing.com calendar fetch failed: %s", e)
        return []

    html_data = result.get("data", "")
    if not html_data:
        logger.warning("Investing.com returned empty calendar data")
        return []

    soup = BeautifulSoup(html_data, "html.parser")
    rows = soup.find_all(
        "tr", attrs={"id": lambda x: x and x.startswith("eventRow") if x else False}
    )

    events: List[CalendarEvent] = []

    for tr in rows:
        try:
            # Currency filter
            flag_cell = tr.find("td", class_="flagCur")
            currency = flag_cell.get_text(strip=True) if flag_cell else ""
            if currency not in RELEVANT_CURRENCIES:
                continue

            # Event ID
            event_id = tr.get("id", "").replace("eventRowId_", "")

            # DateTime (UTC)
            dt_str = tr.get("data-event-datetime", "")
            # Convert "2026/04/16 12:30:00" → "2026-04-16T12:30:00"
            dt_iso = dt_str.replace("/", "-").replace(" ", "T") if dt_str else ""

            # Impact (count filled bull icons)
            impact_cell = tr.find("td", class_="sentiment")
            impact_stars = 0
            if impact_cell:
                bulls = impact_cell.find_all(
                    "i",
                    class_=lambda c: c and "grayFullBullishIcon" in c if c else False,
                )
                impact_stars = len(bulls)
            impact = IMPACT_MAP.get(impact_stars, "low")

            # Event name
            event_cell = tr.find("td", class_="event")
            event_name = event_cell.get_text(strip=True) if event_cell else ""
            if not event_name:
                continue

            # Actual / Forecast / Previous
            actual = (tr.find("td", class_="act") or {})
            forecast = (tr.find("td", class_="fore") or {})
            previous = (tr.find("td", class_="prev") or {})
            actual_str = actual.get_text(strip=True) if hasattr(actual, "get_text") else ""
            forecast_str = forecast.get_text(strip=True) if hasattr(forecast, "get_text") else ""
            previous_str = previous.get_text(strip=True) if hasattr(previous, "get_text") else ""

            # Determine which instruments this affects
            affects: List[str] = []
            if currency == "USD":
                affects.append("MES")
            if currency == "BRL":
                affects.append("WDO")
            # High-impact USD events also affect WDO (global risk)
            if currency == "USD" and impact == "high":
                affects.append("WDO")
            affects.append("ALL")

            # Blackout windows
            before, after = BLACKOUT_DEFAULTS.get(impact, (5, 5))
            name_lower = event_name.lower()
            for keyword, (kb, ka) in WIDE_BLACKOUT_KEYWORDS.items():
                if keyword in name_lower:
                    before = max(before, kb)
                    after = max(after, ka)
                    break

            events.append(CalendarEvent(
                event_id=event_id,
                name=event_name,
                datetime_utc=dt_iso,
                currency=currency,
                impact=impact,
                impact_stars=impact_stars,
                actual=actual_str,
                forecast=forecast_str,
                previous=previous_str,
                affects=affects,
                blackout_before_min=before,
                blackout_after_min=after,
            ))

        except Exception as e:
            logger.debug("Failed to parse calendar row: %s", e)
            continue

    logger.info(
        "Investing.com calendar: %d events (USD: %d, BRL: %d)",
        len(events),
        sum(1 for e in events if e.currency == "USD"),
        sum(1 for e in events if e.currency == "BRL"),
    )
    return events


def _read_cache() -> Optional[List[CalendarEvent]]:
    """Read cached calendar if still valid."""
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text())
        cached_at = data.get("_cached_at", 0)
        if time.time() - cached_at > CACHE_TTL:
            return None  # expired
        events = []
        for d in data.get("events", []):
            events.append(CalendarEvent(**{
                k: v for k, v in d.items() if k != "_cached_at"
            }))
        return events
    except Exception as e:
        logger.debug("Calendar cache read failed: %s", e)
        return None


def _write_cache(events: List[CalendarEvent]) -> None:
    """Write events to disk cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "_cached_at": time.time(),
        "events": [e.to_dict() for e in events],
    }
    try:
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning("Calendar cache write failed: %s", e)


def get_calendar_events(force_refresh: bool = False) -> List[CalendarEvent]:
    """Get upcoming economic events (cached, auto-refreshes every 4h).

    Returns list of CalendarEvent for USD + BRL, next 7 days.
    """
    if not force_refresh:
        cached = _read_cache()
        if cached is not None:
            return cached

    events = _fetch_from_investing(days_ahead=7)
    if events:
        _write_cache(events)
    return events


def refresh_calendar() -> List[CalendarEvent]:
    """Force-refresh the calendar from investing.com."""
    return get_calendar_events(force_refresh=True)


def inject_into_calendar_filter() -> int:
    """Load investing.com events into the static CalendarFilter.

    Converts CalendarEvent objects to EconomicEvent objects and adds
    them to the SCHEDULED_EVENTS list in the calendar module.

    Returns number of events injected.
    """
    from takata_engine.utils.calendar import EconomicEvent, SCHEDULED_EVENTS

    events = get_calendar_events()
    if not events:
        return 0

    # Track what we've already injected to avoid duplicates
    existing_keys = {(d, e.name) for d, e in SCHEDULED_EVENTS}
    injected = 0

    for ev in events:
        if not ev.datetime_utc:
            continue

        try:
            dt = datetime.fromisoformat(ev.datetime_utc)
            date_str = dt.strftime("%Y-%m-%d")
            event_time = dt_time(dt.hour, dt.minute)
        except (ValueError, TypeError):
            continue

        # Skip if already in the static list
        if (date_str, ev.name) in existing_keys:
            continue

        # Only inject medium + high impact (1-star events are noise)
        if ev.impact_stars < 2:
            continue

        eco_event = EconomicEvent(
            name=ev.name,
            time_utc=event_time,
            impact=ev.impact,
            affects=ev.affects,
            blackout_before_min=ev.blackout_before_min,
            blackout_after_min=ev.blackout_after_min,
        )

        SCHEDULED_EVENTS.append((date_str, eco_event))
        existing_keys.add((date_str, ev.name))
        injected += 1

    if injected > 0:
        logger.info("Calendar: injected %d events from investing.com", injected)

    return injected


def get_events_summary() -> Dict[str, Any]:
    """Summary for API/frontend consumption."""
    events = get_calendar_events()
    now = datetime.utcnow()

    # Split by day
    by_day: Dict[str, List[Dict]] = {}
    upcoming_high: List[Dict] = []

    for ev in events:
        d = ev.to_dict()
        day = ev.datetime_utc[:10] if ev.datetime_utc else "unknown"
        by_day.setdefault(day, []).append(d)

        # Track upcoming high-impact events
        if ev.impact_stars >= 3:
            try:
                evt_dt = datetime.fromisoformat(ev.datetime_utc)
                if evt_dt > now:
                    d["minutes_until"] = int((evt_dt - now).total_seconds() / 60)
                    upcoming_high.append(d)
            except (ValueError, TypeError):
                pass

    return {
        "total_events": len(events),
        "usd_events": sum(1 for e in events if e.currency == "USD"),
        "brl_events": sum(1 for e in events if e.currency == "BRL"),
        "high_impact": sum(1 for e in events if e.impact_stars >= 3),
        "medium_impact": sum(1 for e in events if e.impact_stars == 2),
        "upcoming_high_impact": upcoming_high[:10],
        "by_day": {day: evts for day, evts in sorted(by_day.items())},
    }
