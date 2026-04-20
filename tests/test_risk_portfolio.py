"""Unit tests for portfolio risk module."""

import numpy as np
import pandas as pd
import pytest

from takata_core.risk_portfolio.analytics import (
    annualized_volatility,
    beta,
    cvar,
    max_drawdown_analysis,
    sharpe_ratio,
    var_historical,
)
from takata_core.risk_portfolio.concentration import (
    herfindahl_index,
    top_holdings_concentration,
)
from takata_core.risk_portfolio.hedging import recommend_hedges
from takata_core.risk_portfolio.portfolio import Holding
from takata_core.risk_portfolio.stress import run_scenario, SCENARIOS


class TestAnalytics:
    @pytest.fixture
    def returns(self):
        rng = np.random.default_rng(42)
        return pd.Series(rng.normal(0.0005, 0.01, 252), name="portfolio")

    @pytest.fixture
    def benchmark(self):
        rng = np.random.default_rng(99)
        return pd.Series(rng.normal(0.0004, 0.012, 252), name="SPY")

    def test_annualized_vol(self, returns):
        vol = annualized_volatility(returns)
        assert 0.05 < vol < 0.30  # reasonable range

    def test_beta_near_one_for_similar(self, returns):
        b = beta(returns, returns * 1.1 + 0.0001)
        assert 0.5 < b < 2.0

    def test_var_positive(self, returns):
        v = var_historical(returns, 0.95)
        assert v > 0

    def test_cvar_greater_than_var(self, returns):
        v = var_historical(returns, 0.95)
        cv = cvar(returns, 0.95)
        assert cv >= v

    def test_max_drawdown(self, returns):
        result = max_drawdown_analysis(returns)
        assert result["max_drawdown"] > 0
        assert result["peak_date"] is not None

    def test_sharpe(self, returns):
        s = sharpe_ratio(returns)
        assert isinstance(s, float)


class TestConcentration:
    def test_herfindahl_equal_weights(self):
        holdings = [
            Holding("A", 100, 10, 10, 1000, 0.25),
            Holding("B", 100, 10, 10, 1000, 0.25),
            Holding("C", 100, 10, 10, 1000, 0.25),
            Holding("D", 100, 10, 10, 1000, 0.25),
        ]
        hhi = herfindahl_index(holdings)
        assert abs(hhi - 0.25) < 0.001  # 4 equal weights: 4 * 0.25^2 = 0.25

    def test_herfindahl_concentrated(self):
        holdings = [
            Holding("A", 100, 10, 10, 9000, 0.90),
            Holding("B", 100, 10, 10, 1000, 0.10),
        ]
        hhi = herfindahl_index(holdings)
        assert hhi > 0.80  # very concentrated

    def test_top5_concentration(self):
        holdings = [
            Holding(f"T{i}", 100, 10, 10, 1000, 0.1) for i in range(10)
        ]
        assert abs(top_holdings_concentration(holdings, 5) - 0.5) < 0.001


class TestStress:
    def test_recession_scenario(self):
        h1 = Holding("SPY", 100, 450)
        h1.current_price = 500; h1.market_value = 50000; h1.weight = 0.83
        h2 = Holding("TLT", 100, 95)
        h2.current_price = 100; h2.market_value = 10000; h2.weight = 0.17
        holdings = [h1, h2]

        result = run_scenario(holdings, 60000, "recession")
        assert result.name == "Recession"
        assert result.portfolio_impact_pct < 0  # should be negative

    def test_all_scenarios_defined(self):
        assert len(SCENARIOS) >= 5


class TestHedging:
    def test_high_beta_recommendation(self):
        recs = recommend_hedges(
            risk_metrics={"annualized_volatility": 0.25, "var_95": 0.03, "cvar_95": 0.04,
                          "max_drawdown": {"max_drawdown": 0.20}},
            sector_weights={"Technology": 0.40},
            asset_class_weights={"US Equity": 0.90},
            beta=1.5,
            portfolio_value=100000,
        )
        assert len(recs) > 0
        instruments = [r.instrument for r in recs]
        # Should recommend something for high beta
        assert any("put" in i.lower() or "vix" in i.lower() or "uvxy" in i.lower() for i in instruments)

    def test_no_bonds_recommendation(self):
        recs = recommend_hedges(
            risk_metrics={"annualized_volatility": 0.15, "var_95": 0.02, "cvar_95": 0.03,
                          "max_drawdown": {"max_drawdown": 0.10}},
            sector_weights={},
            asset_class_weights={"US Equity": 1.0},
            beta=1.0,
            portfolio_value=50000,
        )
        instruments = " ".join(r.instrument for r in recs).lower()
        assert "agg" in instruments or "bnd" in instruments or "tlt" in instruments
