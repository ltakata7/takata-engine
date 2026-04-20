"""Unit tests for session definitions."""

import pandas as pd
import pytest

from takata_engine.utils.session import CME_RTH, CME_ETH, B3_REGULAR, get_session


class TestCMERTH:
    def test_during_session(self):
        ts = pd.Timestamp("2025-01-02 10:00", tz="America/New_York")
        assert CME_RTH.is_active(ts) is True

    def test_before_session(self):
        ts = pd.Timestamp("2025-01-02 09:00", tz="America/New_York")
        assert CME_RTH.is_active(ts) is False

    def test_after_session(self):
        ts = pd.Timestamp("2025-01-02 16:30", tz="America/New_York")
        assert CME_RTH.is_active(ts) is False


class TestCMEETH:
    def test_evening(self):
        ts = pd.Timestamp("2025-01-02 20:00", tz="America/New_York")
        assert CME_ETH.is_active(ts) is True

    def test_early_morning(self):
        ts = pd.Timestamp("2025-01-02 03:00", tz="America/New_York")
        assert CME_ETH.is_active(ts) is True


class TestGetSession:
    def test_mes(self):
        assert get_session("MES").exchange == "CME"

    def test_wdo(self):
        assert get_session("WDO").exchange == "B3"

    def test_unknown(self):
        with pytest.raises(ValueError):
            get_session("UNKNOWN")
