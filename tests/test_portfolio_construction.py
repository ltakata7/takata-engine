"""Unit tests for portfolio construction module."""

import pytest

from takata_core.portfolio.construction import PortfolioConstructor, _determine_risk_profile


class TestRiskProfile:
    def test_young_aggressive(self):
        assert _determine_risk_profile(25, "high") == "aggressive"

    def test_old_conservative(self):
        assert _determine_risk_profile(65, "low") == "very_conservative"

    def test_middle_moderate(self):
        assert _determine_risk_profile(45, "moderate") == "moderate"

    def test_explicit_profile(self):
        assert _determine_risk_profile(30, "conservative") == "conservative"


class TestPortfolioConstructor:
    def test_allocations_sum_to_100(self):
        builder = PortfolioConstructor(age=35, amount=100000, risk="moderate")
        plan = builder.build()
        total = sum(a.target_pct for a in plan.allocations)
        assert abs(total - 100.0) < 1.0

    def test_aggressive_has_more_equity(self):
        agg = PortfolioConstructor(age=25, amount=50000, risk="aggressive").build()
        con = PortfolioConstructor(age=60, amount=50000, risk="conservative").build()
        agg_eq = sum(a.target_pct for a in agg.allocations if "Equity" in a.category or "Emerging" in a.category)
        con_eq = sum(a.target_pct for a in con.allocations if "Equity" in a.category or "Emerging" in a.category)
        assert agg_eq > con_eq

    def test_has_core_and_satellite(self):
        plan = PortfolioConstructor(age=40, amount=100000, risk="moderate").build()
        roles = {a.role for a in plan.allocations}
        assert "core" in roles

    def test_dca_plan_exists(self):
        plan = PortfolioConstructor(age=35, amount=120000, risk="growth").build()
        assert "$10,000/month" in plan.dca_plan

    def test_expected_returns(self):
        plan = PortfolioConstructor(age=35, amount=100000, risk="moderate").build()
        assert plan.expected_return_low < plan.expected_return_high
        assert plan.expected_return_low > 0
