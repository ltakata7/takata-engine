"""Unit tests for backtester."""

import pytest

from takata_core.backtest import Backtester, BacktestResult


class TestBacktester:
    def test_run_produces_result(self):
        bt = Backtester(
            instrument="MES",
            feed_path="sample_data/mes_5min.csv",
            initial_capital=10000.0,
        )
        result = bt.run()
        assert isinstance(result, BacktestResult)
        assert result.metrics["total_trades"] > 0
        assert len(result.equity) > 0

    def test_run_with_overrides(self):
        bt = Backtester(
            instrument="MES",
            feed_path="sample_data/mes_5min.csv",
        )
        result = bt.run(config_overrides={"risk.stop_atr_mult": 1.0})
        assert isinstance(result, BacktestResult)
        assert result.config_overrides == {"risk.stop_atr_mult": 1.0}

    def test_report_string(self):
        bt = Backtester(
            instrument="MES",
            feed_path="sample_data/mes_5min.csv",
        )
        result = bt.run()
        report = result.report()
        assert "Sharpe ratio" in report

    def test_sweep_returns_sorted(self):
        bt = Backtester(
            instrument="MES",
            feed_path="sample_data/mes_5min.csv",
        )
        results = bt.run_sweep({
            "risk.stop_atr_mult": [1.0, 2.0],
        })
        assert len(results) == 2
        # Sorted by Sharpe descending
        assert results[0].sharpe >= results[1].sharpe
