"""Market session definitions for CME and B3 exchanges."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

import pandas as pd


@dataclass(frozen=True)
class Session:
    """Trading session definition."""
    name: str
    exchange: str
    open_time: time
    close_time: time
    timezone: str
    crosses_midnight: bool = False

    def is_active(self, ts: datetime | pd.Timestamp) -> bool:
        """Check if a timestamp falls within this session."""
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(self.timezone)
        else:
            ts = ts.tz_convert(self.timezone)

        t = ts.time()
        if self.crosses_midnight:
            return t >= self.open_time or t < self.close_time
        return self.open_time <= t < self.close_time

    def session_open(self, ts: datetime | pd.Timestamp) -> pd.Timestamp:
        """Return the session open timestamp for the date of *ts*.

        Useful for VWAP reset.
        """
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(self.timezone)
        else:
            ts = ts.tz_convert(self.timezone)

        dt = ts.normalize() + pd.Timedelta(
            hours=self.open_time.hour, minutes=self.open_time.minute
        )
        return dt.tz_localize(self.timezone) if dt.tzinfo is None else dt


# ── Pre-defined sessions ────────────────────────────────────────────

CME_RTH = Session(
    name="CME Regular Trading Hours",
    exchange="CME",
    open_time=time(9, 30),
    close_time=time(16, 0),
    timezone="America/New_York",
)

CME_ETH = Session(
    name="CME Extended Trading Hours",
    exchange="CME",
    open_time=time(18, 0),
    close_time=time(17, 0),
    timezone="America/New_York",
    crosses_midnight=True,
)

B3_REGULAR = Session(
    name="B3 Regular Session",
    exchange="B3",
    open_time=time(9, 0),
    close_time=time(17, 55),
    timezone="America/Sao_Paulo",
)

B3_AFTER_MARKET = Session(
    name="B3 After Market",
    exchange="B3",
    open_time=time(18, 25),
    close_time=time(18, 45),
    timezone="America/Sao_Paulo",
)

SESSIONS = {
    "CME_RTH": CME_RTH,
    "CME_ETH": CME_ETH,
    "B3_REGULAR": B3_REGULAR,
    "B3_AFTER_MARKET": B3_AFTER_MARKET,
}


def get_session(instrument: str) -> Session:
    """Return the primary trading session for an instrument."""
    instrument = instrument.upper()
    if instrument in ("MES", "ES", "NQ", "MNQ"):
        return CME_RTH
    if instrument in ("WDO", "WIN", "IND", "DOL"):
        return B3_REGULAR
    raise ValueError(f"Unknown instrument: {instrument}")


def get_wdo_session_phase(ts: datetime | pd.Timestamp | None = None) -> str:
    """Determine the WDO session phase based on PTAX consultation windows.

    BCB consults dealers at 4 random times within these windows (BRT):
    - 10:00-10:10 — First consultation
    - 11:00-11:10 — Second consultation
    - 12:00-12:10 — Third consultation
    - 13:00-13:10 — Fourth (final) consultation

    Returns
    -------
    str
        ``"pre_ptax"`` — 09:00-10:00: follow overnight, cautious
        ``"ptax_formation"`` — 10:00-13:10: follow order flow, VWAP critical
        ``"ptax_window_1"`` through ``"ptax_window_4"`` — during actual windows
        ``"post_ptax"`` — 13:10-17:55: technical trading, regime reliable
    """
    if ts is None:
        from datetime import datetime as dt
        ts = pd.Timestamp(dt.now(tz=pd.Timestamp.now(tz="America/Sao_Paulo").tzinfo))
    else:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("America/Sao_Paulo")
        else:
            ts = ts.tz_convert("America/Sao_Paulo")

    h, m = ts.hour, ts.minute

    if h < 9:
        return "pre_market"
    elif h == 9:
        return "pre_ptax"
    elif h == 10 and m <= 10:
        return "ptax_window_1"
    elif h == 11 and m <= 10:
        return "ptax_window_2"
    elif h == 12 and m <= 10:
        return "ptax_window_3"
    elif h == 13 and m <= 10:
        return "ptax_window_4"
    elif h < 13 or (h == 13 and m <= 10):
        return "ptax_formation"
    elif h < 18:
        return "post_ptax"
    else:
        return "after_market"


def get_mes_session_phase(ts: datetime | pd.Timestamp | None = None) -> str:
    """Determine MES session phase (ET).

    Returns
    -------
    str
        Session phase name.
    """
    if ts is None:
        from datetime import datetime as dt
        ts = pd.Timestamp(dt.now(tz=pd.Timestamp.now(tz="America/New_York").tzinfo))
    else:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("America/New_York")
        else:
            ts = ts.tz_convert("America/New_York")

    h, m = ts.hour, ts.minute

    if h < 4:
        return "overnight"
    elif h < 9 or (h == 9 and m < 30):
        return "pre_market"
    elif h == 9 and m < 45:
        return "open"
    elif h < 11 or (h == 11 and m < 30):
        return "morning"
    elif h < 14:
        return "midday"
    elif h < 15 or (h == 15 and m < 30):
        return "afternoon"
    elif h < 16:
        return "close"
    else:
        return "after_hours"
