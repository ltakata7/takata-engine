"""Unit tests for signal generator and position management."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from takata_engine.signals.position import Position, PositionManager


def _make_bar(open_=100, high=105, low=95, close=102, volume=1000):
    return pd.Series({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


class TestPosition:
    def test_close_long_profit(self):
        pos = Position("MES", "long", 100.0, datetime(2025, 1, 2), 1.0, 95.0, 110.0, "calm")
        pos.close(110.0, datetime(2025, 1, 2, 10), "target_hit")
        assert pos.pnl == 10.0
        assert pos.status == "closed"

    def test_close_short_profit(self):
        pos = Position("MES", "short", 100.0, datetime(2025, 1, 2), 1.0, 105.0, 90.0, "calm")
        pos.close(90.0, datetime(2025, 1, 2, 10), "target_hit")
        assert pos.pnl == 10.0

    def test_close_long_loss(self):
        pos = Position("MES", "long", 100.0, datetime(2025, 1, 2), 1.0, 95.0, 110.0, "calm")
        pos.close(95.0, datetime(2025, 1, 2, 10), "stop_hit")
        assert pos.pnl == -5.0

    def test_check_stop_long(self):
        pos = Position("MES", "long", 100.0, datetime(2025, 1, 2), 1.0, 95.0, 110.0, "calm")
        bar_hit = _make_bar(low=94.0)
        bar_miss = _make_bar(low=96.0)
        assert pos.check_stop(bar_hit) == True
        assert pos.check_stop(bar_miss) == False

    def test_check_target_long(self):
        pos = Position("MES", "long", 100.0, datetime(2025, 1, 2), 1.0, 95.0, 110.0, "calm")
        bar_hit = _make_bar(high=111.0, low=99.0)   # low above stop so only target triggers
        bar_miss = _make_bar(high=109.0, low=99.0)
        assert pos.check_target(bar_hit) == True
        assert pos.check_target(bar_miss) == False

    def test_unrealized_pnl(self):
        pos = Position("MES", "long", 100.0, datetime(2025, 1, 2), 2.0, 95.0, 110.0, "calm")
        assert pos.unrealized_pnl(105.0) == 10.0


class TestPositionManager:
    def test_open_and_close(self):
        mgr = PositionManager(point_value=5.0)
        mgr.open_position("MES", "long", 100.0, datetime(2025, 1, 2), 1.0, 95.0, 110.0, "calm")
        assert mgr.has_open("MES")

        bar = _make_bar(low=94.0)  # stop hit
        closed = mgr.check_exits("MES", bar, datetime(2025, 1, 2, 10))
        assert closed is not None
        assert closed.exit_reason == "stop_hit"
        assert closed.pnl == -5.0 * 5.0  # -25.0 in currency
        assert not mgr.has_open("MES")

    def test_target_hit(self):
        mgr = PositionManager(point_value=5.0)
        mgr.open_position("MES", "long", 100.0, datetime(2025, 1, 2), 1.0, 95.0, 110.0, "calm")
        bar = _make_bar(high=111.0, low=99.0)  # low above stop
        closed = mgr.check_exits("MES", bar, datetime(2025, 1, 2, 10))
        assert closed.exit_reason == "target_hit"
        assert closed.pnl == 10.0 * 5.0  # 50.0

    def test_daily_pnl(self):
        mgr = PositionManager(point_value=1.0)
        mgr.open_position("MES", "long", 100.0, datetime(2025, 1, 2, 9, 30), 1.0, 95.0, 110.0, "calm")
        mgr.close_position("MES", 105.0, datetime(2025, 1, 2, 10, 0), "manual")
        assert mgr.daily_pnl() == 5.0

    def test_replace_existing_position(self):
        mgr = PositionManager(point_value=1.0)
        mgr.open_position("MES", "long", 100.0, datetime(2025, 1, 2, 9, 30), 1.0, 95.0, 110.0, "calm")
        # Opening a new position replaces the old one
        mgr.open_position("MES", "short", 102.0, datetime(2025, 1, 2, 10, 0), 1.0, 107.0, 97.0, "calm")
        assert mgr.get_open("MES").side == "short"
        assert mgr.trade_count == 1  # old position was closed
