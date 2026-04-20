"""Signal filters — session time gates and calendar blackouts."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from takata_core.utils.session import Session, get_session


class SignalFilter:
    """Gates signal generation by trading hours and time-of-day rules.

    Parameters
    ----------
    session : Session
        Primary trading session.
    session_only : bool
        If ``True``, only allow signals during the session.
    first_n_minutes_skip : int
        Skip first N minutes after session open.
    last_n_minutes_skip : int
        Skip last N minutes before session close.
    """

    def __init__(
        self,
        session: Session,
        session_only: bool = True,
        first_n_minutes_skip: int = 15,
        last_n_minutes_skip: int = 15,
    ):
        self.session = session
        self.session_only = session_only
        self.first_n_minutes_skip = first_n_minutes_skip
        self.last_n_minutes_skip = last_n_minutes_skip

    @classmethod
    def from_config(cls, instrument: str, config: Dict[str, Any]) -> "SignalFilter":
        """Construct from instrument name and filters config section."""
        session = get_session(instrument)
        filters_cfg = config.get("filters", {})
        return cls(
            session=session,
            session_only=filters_cfg.get("session_only", True),
            first_n_minutes_skip=filters_cfg.get("first_n_minutes_skip", 15),
            last_n_minutes_skip=filters_cfg.get("last_n_minutes_skip", 15),
        )

    def is_allowed(self, ts: datetime | pd.Timestamp) -> bool:
        """Check if signal generation is allowed at this timestamp.

        Returns
        -------
        bool
            ``True`` if signals are allowed.
        """
        ts = pd.Timestamp(ts)

        # Session check
        if self.session_only and not self.session.is_active(ts):
            return False

        # Convert to session timezone for time-of-day checks
        if ts.tzinfo is None:
            ts_local = ts.tz_localize(self.session.timezone)
        else:
            ts_local = ts.tz_convert(self.session.timezone)

        t = ts_local.time()

        # Skip first N minutes after open
        if self.first_n_minutes_skip > 0:
            open_dt = datetime.combine(datetime.min, self.session.open_time)
            skip_until = (open_dt + timedelta(minutes=self.first_n_minutes_skip)).time()
            if self.session.open_time <= t < skip_until:
                return False

        # Skip last N minutes before close
        if self.last_n_minutes_skip > 0:
            close_dt = datetime.combine(datetime.min, self.session.close_time)
            skip_from = (close_dt - timedelta(minutes=self.last_n_minutes_skip)).time()
            if not self.session.crosses_midnight and skip_from <= t < self.session.close_time:
                return False

        return True
