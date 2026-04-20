"""Unit tests for risk manager."""

import pytest

from takata_core.signals.risk_manager import RiskManager


@pytest.fixture
def risk_mgr():
    return RiskManager(
        config={
            "max_position_size": 10.0,
            "stop_atr_mult": 2.0,
            "target_atr_mult": 3.0,
            "max_daily_loss": 500.0,
        },
        point_value=5.0,  # MES
    )


class TestComputeStop:
    def test_long_stop_below_price(self, risk_mgr):
        stop = risk_mgr.compute_stop(5000.0, atr_value=10.0, direction="long")
        assert stop == 4980.0  # 5000 - 2*10

    def test_short_stop_above_price(self, risk_mgr):
        stop = risk_mgr.compute_stop(5000.0, atr_value=10.0, direction="short")
        assert stop == 5020.0  # 5000 + 2*10


class TestComputeTarget:
    def test_long_target_above_price(self, risk_mgr):
        target = risk_mgr.compute_target(5000.0, atr_value=10.0, direction="long")
        assert target == 5030.0  # 5000 + 3*10

    def test_short_target_below_price(self, risk_mgr):
        target = risk_mgr.compute_target(5000.0, atr_value=10.0, direction="short")
        assert target == 4970.0  # 5000 - 3*10


class TestComputeSize:
    def test_basic_sizing(self, risk_mgr):
        # Risk $100, stop 20 points away, point_value=5 → risk_per_unit = 100
        # size = 100 / 100 = 1
        size = risk_mgr.compute_size(price=5000.0, stop=4980.0, account_risk=100.0)
        assert size == 1.0

    def test_capped_by_max(self, risk_mgr):
        # Huge account_risk → would give 100 contracts but capped at 10
        size = risk_mgr.compute_size(price=5000.0, stop=4980.0, account_risk=100000.0)
        assert size == 10.0

    def test_zero_risk(self, risk_mgr):
        size = risk_mgr.compute_size(price=5000.0, stop=5000.0, account_risk=100.0)
        assert size == 0.0


class TestDailyLimit:
    def test_within_limit(self, risk_mgr):
        assert risk_mgr.check_daily_limit(-200.0) is True

    def test_at_limit(self, risk_mgr):
        assert risk_mgr.check_daily_limit(-500.0) is False

    def test_beyond_limit(self, risk_mgr):
        assert risk_mgr.check_daily_limit(-600.0) is False

    def test_positive_pnl(self, risk_mgr):
        assert risk_mgr.check_daily_limit(1000.0) is True


class TestUpdateFromRegime:
    def test_updates_params(self, risk_mgr):
        risk_mgr.update_from_regime({
            "stop_atr_mult": 1.0,
            "target_atr_mult": 1.5,
            "max_position_size": 5.0,
        })
        assert risk_mgr.stop_atr_mult == 1.0
        assert risk_mgr.target_atr_mult == 1.5
        assert risk_mgr.max_position_size == 5.0
