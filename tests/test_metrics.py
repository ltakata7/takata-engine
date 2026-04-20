"""Unit tests for backtest performance metrics."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from takata_engine.backtest.metrics import (
    avg_loser,
    avg_winner,
    calmar_ratio,
    compute_all,
    equity_curve,
    expectancy,
    format_report,
    max_consecutive_losses,
    max_drawdown,
    max_drawdown_duration,
    profit_factor,
    returns_series,
    sharpe_ratio,
    sortino_ratio,
    trade_duration_stats,
    win_rate,
)
from takata_engine.signals.position import Position


def _pos(pnl: float, minutes: int = 30, side: str = "long") -> Position:
    """Helper to create a closed Position with given P&L."""
    t0 = datetime(2025, 1, 2, 10, 0)
    p = Position("MES", side, 100.0, t0, 1.0, 95.0, 110.0, "calm")
    p.close(100.0 + pnl if side == "long" else 100.0 - pnl,
            t0 + timedelta(minutes=minutes), "test")
    p.pnl = pnl  # override for precision
    return p


class TestTradeMetrics:
    def test_win_rate(self):
        positions = [_pos(10), _pos(-5), _pos(20), _pos(-3)]
        assert win_rate(positions) == 0.5

    def test_win_rate_empty(self):
        assert win_rate([]) == 0.0

    def test_profit_factor(self):
        positions = [_pos(10), _pos(-5), _pos(20), _pos(-5)]
        assert profit_factor(positions) == 3.0  # 30 / 10

    def test_profit_factor_no_losses(self):
        positions = [_pos(10), _pos(20)]
        assert profit_factor(positions) == float("inf")

    def test_expectancy(self):
        positions = [_pos(10), _pos(-5), _pos(20), _pos(-5)]
        assert expectancy(positions) == 5.0  # 20 / 4

    def test_avg_winner_loser(self):
        positions = [_pos(10), _pos(-5), _pos(20), _pos(-3)]
        assert avg_winner(positions) == 15.0
        assert avg_loser(positions) == -4.0

    def test_max_consecutive_losses(self):
        positions = [_pos(10), _pos(-5), _pos(-3), _pos(-2), _pos(10), _pos(-1)]
        assert max_consecutive_losses(positions) == 3

    def test_trade_duration(self):
        positions = [_pos(10, minutes=30), _pos(-5, minutes=60)]
        stats = trade_duration_stats(positions)
        assert stats["avg_minutes"] == 45.0
        assert stats["max_minutes"] == 60.0


class TestEquityCurveMetrics:
    def test_equity_curve_shape(self):
        positions = [_pos(10), _pos(-5), _pos(20)]
        eq = equity_curve(positions, 10000.0)
        assert len(eq) == 3
        assert eq.iloc[0] == 10010.0
        assert eq.iloc[-1] == 10025.0

    def test_sharpe_positive(self):
        # Consistently positive returns
        rets = pd.Series([0.01, 0.02, 0.01, 0.015, 0.01, 0.02])
        s = sharpe_ratio(rets, trades_per_year=252)
        assert s > 0

    def test_sharpe_zero_std(self):
        rets = pd.Series([0.01, 0.01, 0.01])
        assert sharpe_ratio(rets) == 0.0

    def test_sortino_ignores_upside(self):
        # All positive returns → infinite Sortino
        rets = pd.Series([0.01, 0.02, 0.03])
        s = sortino_ratio(rets)
        assert s == float("inf")

    def test_max_drawdown(self):
        eq = pd.Series([100, 110, 105, 90, 95, 100])
        dd = max_drawdown(eq)
        # Peak was 110, trough was 90 → dd = 20/110 ≈ 0.1818
        assert abs(dd - 20.0 / 110) < 0.001

    def test_max_drawdown_duration(self):
        eq = pd.Series([100, 110, 105, 90, 95, 100, 112])
        dur = max_drawdown_duration(eq)
        # In drawdown from index 2 through 5 (4 trades)
        assert dur == 4


class TestComputeAll:
    def test_returns_all_keys(self):
        positions = [_pos(10), _pos(-5), _pos(20)]
        m = compute_all(positions, initial_capital=10000)
        assert "total_trades" in m
        assert "sharpe_ratio" in m
        assert "max_drawdown" in m
        assert "trade_duration" in m
        assert m["total_trades"] == 3


class TestFormatReport:
    def test_produces_string(self):
        positions = [_pos(10), _pos(-5)]
        m = compute_all(positions, initial_capital=10000)
        report = format_report(m)
        assert "Sharpe ratio" in report
        assert "Win rate" in report
