"""Economic calendar — event-based signal gating.

Defines high-impact events that should block or reduce signal generation.
Maintains a static schedule of recurring events + ability to add ad-hoc events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EconomicEvent:
    """A scheduled economic event that affects trading."""
    name: str
    time_utc: Optional[time]  # None = all day
    impact: str  # "high", "medium", "low"
    affects: List[str]  # ["MES", "WDO", "ALL"]
    blackout_before_min: int = 15  # minutes before to stop trading
    blackout_after_min: int = 10   # minutes after to resume
    reduce_size_pct: float = 0.0   # 0 = full blackout, 0.5 = half size
    # When used in RECURRING_WEEKLY, restricts firing to the first occurrence
    # of its weekday in the month (e.g. NFP = first Friday).
    first_of_month_only: bool = False


# ── Recurring weekly/monthly events (US + Brazil) ──

# Day of week: 0=Monday ... 4=Friday
RECURRING_WEEKLY = {
    # US events
    4: [  # Friday
        # NFP: first Friday of the month only
        EconomicEvent("NFP / Nonfarm Payrolls", time(12, 30), "high", ["MES", "ALL"], 30, 30,
                      first_of_month_only=True),
    ],
}

# Day of month events
RECURRING_MONTHLY: List[Tuple[str, EconomicEvent]] = []  # populated dynamically

# ── Known high-impact events (static schedule) ──
# These are the events that ALWAYS cause major volatility

HIGH_IMPACT_US = [
    # Fed
    EconomicEvent("FOMC Rate Decision", time(18, 0), "high", ["MES", "ALL"], 30, 60),
    EconomicEvent("FOMC Minutes", time(18, 0), "high", ["MES"], 15, 30),
    EconomicEvent("Fed Chair Speech", None, "high", ["MES", "ALL"], 15, 15),
    # Data releases (all at 8:30 ET = 12:30 UTC usually)
    EconomicEvent("CPI / Consumer Price Index", time(12, 30), "high", ["MES", "ALL"], 15, 15),
    EconomicEvent("PPI / Producer Price Index", time(12, 30), "high", ["MES"], 10, 10),
    EconomicEvent("Retail Sales", time(12, 30), "medium", ["MES"], 10, 10),
    EconomicEvent("GDP", time(12, 30), "high", ["MES", "ALL"], 15, 15),
    EconomicEvent("PCE / Core PCE", time(12, 30), "high", ["MES"], 15, 15),
    EconomicEvent("ISM Manufacturing", time(14, 0), "medium", ["MES"], 10, 10),
    EconomicEvent("Initial Jobless Claims", time(12, 30), "low", ["MES"], 5, 5),
]

HIGH_IMPACT_BRAZIL = [
    # BCB
    EconomicEvent("Selic Rate Decision (COPOM)", time(21, 30), "high", ["WDO", "ALL"], 60, 60),
    EconomicEvent("COPOM Minutes", time(11, 0), "high", ["WDO"], 15, 30),
    EconomicEvent("BCB Intervention / Swap Auction", None, "high", ["WDO"], 5, 15),
    # PTAX windows (already handled in session phases, but blackout for safety)
    EconomicEvent("PTAX Window 4 (Final)", time(16, 0), "high", ["WDO"], 5, 5),  # 13:00 BRT = 16:00 UTC
    # Data
    EconomicEvent("IPCA (Brazil CPI)", time(12, 0), "high", ["WDO"], 15, 15),
    EconomicEvent("Brazil GDP", time(12, 0), "high", ["WDO"], 15, 15),
    EconomicEvent("Brazil PMI", time(13, 0), "medium", ["WDO"], 10, 10),
    EconomicEvent("Brazil Trade Balance", time(18, 0), "medium", ["WDO"], 10, 10),
    EconomicEvent("B3 Foreign Flow Data", time(21, 0), "medium", ["WDO"], 5, 5),
]

# ── Scheduled events with known dates ──
# Format: (date_str, event) — automatically loaded for the matching day

SCHEDULED_EVENTS: List[Tuple[str, EconomicEvent]] = [
    # --- April 2026 ---
    # Week of April 6-10
    ("2026-04-07", EconomicEvent("Consumer Credit", time(19, 0), "low", ["MES"], 5, 5)),
    ("2026-04-08", EconomicEvent("NFIB Small Business", time(10, 0), "low", ["MES"], 5, 5)),
    ("2026-04-08", EconomicEvent("Brazil IPCA-15 (Preview CPI)", time(12, 0), "high", ["WDO"], 15, 15)),
    ("2026-04-09", EconomicEvent("FOMC Minutes (March Meeting)", time(18, 0), "high", ["MES", "ALL"], 15, 30)),
    ("2026-04-10", EconomicEvent("CPI / Consumer Price Index", time(12, 30), "high", ["MES", "ALL"], 15, 15)),
    ("2026-04-10", EconomicEvent("Initial Jobless Claims", time(12, 30), "low", ["MES"], 5, 5)),
    # Week of April 13-17
    ("2026-04-14", EconomicEvent("PPI / Producer Price Index", time(12, 30), "high", ["MES"], 10, 10)),
    ("2026-04-15", EconomicEvent("Retail Sales", time(12, 30), "medium", ["MES"], 10, 10)),
    ("2026-04-15", EconomicEvent("Empire State Manufacturing", time(12, 30), "low", ["MES"], 5, 5)),
    ("2026-04-16", EconomicEvent("Initial Jobless Claims", time(12, 30), "low", ["MES"], 5, 5)),
    ("2026-04-16", EconomicEvent("Housing Starts", time(12, 30), "low", ["MES"], 5, 5)),
    ("2026-04-03", EconomicEvent("Good Friday (US Markets Closed)", None, "high", ["MES", "ALL"], 0, 0)),
]

# ── Ad-hoc events (added at runtime) ──
_adhoc_events: List[Tuple[datetime, EconomicEvent]] = []


def add_event(dt: datetime, event: EconomicEvent) -> None:
    """Add an ad-hoc event (e.g., from news feed)."""
    _adhoc_events.append((dt, event))
    logger.info("Calendar: added %s at %s", event.name, dt)


class CalendarFilter:
    """Checks if trading is allowed based on the economic calendar.

    Parameters
    ----------
    instrument : str
        ``"MES"`` or ``"WDO"``.
    """

    def __init__(self, instrument: str = "MES"):
        self.instrument = instrument.upper()
        self._today_events: List[Tuple[datetime, EconomicEvent]] = []
        self._last_date: Optional[str] = None

    def _load_today_events(self, now: datetime) -> None:
        """Build today's event list."""
        today_str = now.strftime("%Y-%m-%d")
        if self._last_date == today_str:
            return

        self._today_events.clear()

        # Recurring weekly events
        dow = now.weekday()
        for event in RECURRING_WEEKLY.get(dow, []):
            # first_of_month_only gate (e.g. NFP = first Friday of month, day 1-7)
            if event.first_of_month_only and now.day > 7:
                continue
            if self.instrument in event.affects or "ALL" in event.affects:
                if event.time_utc:
                    event_dt = now.replace(hour=event.time_utc.hour, minute=event.time_utc.minute, second=0)
                    self._today_events.append((event_dt, event))

        # Scheduled events (known dates)
        for date_str, event in SCHEDULED_EVENTS:
            if date_str == today_str:
                if self.instrument in event.affects or "ALL" in event.affects:
                    if event.time_utc:
                        event_dt = now.replace(hour=event.time_utc.hour, minute=event.time_utc.minute, second=0)
                    else:
                        # All-day event — block entire day
                        event_dt = now.replace(hour=0, minute=0, second=0)
                        event.blackout_before_min = 1440  # full day
                        event.blackout_after_min = 1440
                    self._today_events.append((event_dt, event))

        # Ad-hoc events (runtime additions)
        for dt, event in _adhoc_events:
            if dt.date() == now.date():
                if self.instrument in event.affects or "ALL" in event.affects:
                    self._today_events.append((dt, event))

        self._last_date = today_str
        if self._today_events:
            logger.info("Calendar [%s]: loaded %d events for %s", self.instrument, len(self._today_events), today_str)

    def check(self, now: Optional[datetime] = None) -> Dict:
        """Check if trading is allowed right now.

        Returns
        -------
        dict
            ``allowed``: bool
            ``reason``: str (if blocked)
            ``reduce_size``: float (1.0 = full, 0.5 = half, 0.0 = no trade)
            ``next_event``: upcoming event name and time
            ``active_blackout``: currently active blackout event
        """
        if now is None:
            now = datetime.utcnow()

        self._load_today_events(now)

        result = {
            "allowed": True,
            "reason": "",
            "reduce_size": 1.0,
            "next_event": None,
            "active_blackout": None,
        }

        nearest_event = None
        nearest_minutes = float("inf")

        for event_dt, event in self._today_events:
            diff = (event_dt - now).total_seconds() / 60  # minutes until event

            # Track nearest upcoming event
            if diff > 0 and diff < nearest_minutes:
                nearest_minutes = diff
                nearest_event = {"name": event.name, "time": str(event_dt.time()), "in_minutes": round(diff)}

            # Check if we're in a blackout window
            if -event.blackout_after_min <= diff <= event.blackout_before_min:
                if event.reduce_size_pct == 0:
                    result["allowed"] = False
                    result["reason"] = f"Blackout: {event.name} ({event.impact} impact)"
                    result["reduce_size"] = 0.0
                else:
                    result["reduce_size"] = min(result["reduce_size"], event.reduce_size_pct)
                    result["reason"] = f"Reduced size: {event.name}"
                result["active_blackout"] = event.name

        result["next_event"] = nearest_event
        return result

    def is_allowed(self, now: Optional[datetime] = None) -> bool:
        """Simple check — True if trading is allowed."""
        return self.check(now)["allowed"]


# Pre-built filters
MES_CALENDAR = CalendarFilter("MES")
WDO_CALENDAR = CalendarFilter("WDO")
