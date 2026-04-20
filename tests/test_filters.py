"""Unit tests for signal filters."""

import pandas as pd
import pytest

from takata_core.signals.filters import SignalFilter
from takata_core.utils.session import CME_RTH


@pytest.fixture
def filter_default():
    return SignalFilter(
        session=CME_RTH,
        session_only=True,
        first_n_minutes_skip=15,
        last_n_minutes_skip=15,
    )


class TestSessionFilter:
    def test_blocks_before_session(self, filter_default):
        ts = pd.Timestamp("2025-01-02 09:00", tz="America/New_York")
        assert filter_default.is_allowed(ts) is False

    def test_blocks_after_session(self, filter_default):
        ts = pd.Timestamp("2025-01-02 16:30", tz="America/New_York")
        assert filter_default.is_allowed(ts) is False

    def test_allows_mid_session(self, filter_default):
        ts = pd.Timestamp("2025-01-02 11:00", tz="America/New_York")
        assert filter_default.is_allowed(ts) is True

    def test_blocks_first_15_minutes(self, filter_default):
        ts = pd.Timestamp("2025-01-02 09:35", tz="America/New_York")
        assert filter_default.is_allowed(ts) is False

    def test_allows_after_first_15(self, filter_default):
        ts = pd.Timestamp("2025-01-02 09:46", tz="America/New_York")
        assert filter_default.is_allowed(ts) is True

    def test_blocks_last_15_minutes(self, filter_default):
        ts = pd.Timestamp("2025-01-02 15:50", tz="America/New_York")
        assert filter_default.is_allowed(ts) is False

    def test_allows_before_last_15(self, filter_default):
        ts = pd.Timestamp("2025-01-02 15:40", tz="America/New_York")
        assert filter_default.is_allowed(ts) is True


class TestFromConfig:
    def test_creates_from_config(self):
        config = {
            "filters": {
                "session_only": True,
                "first_n_minutes_skip": 10,
                "last_n_minutes_skip": 10,
            }
        }
        f = SignalFilter.from_config("MES", config)
        assert f.first_n_minutes_skip == 10
        assert f.session.exchange == "CME"
