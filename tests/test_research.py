"""Unit tests for equity research module."""

import pytest

from takata_engine.research.fundamentals import moat_score, _format_cap


class TestMoatScore:
    def test_high_quality_company(self):
        f = {
            "profit_margin": 0.25, "roe": 0.30, "revenue_growth": 0.15,
            "debt_to_equity": 50, "current_ratio": 2.0, "market_cap": 300e9,
        }
        result = moat_score(f)
        assert result["total"] >= 7.0

    def test_low_quality_company(self):
        f = {
            "profit_margin": 0.02, "roe": 0.03, "revenue_growth": -0.10,
            "debt_to_equity": 300, "current_ratio": 0.5, "market_cap": 1e9,
        }
        result = moat_score(f)
        assert result["total"] < 4.0


class TestFormatCap:
    def test_trillions(self):
        assert _format_cap(3.5e12) == "$3.5T"

    def test_billions(self):
        assert _format_cap(150e9) == "$150.0B"

    def test_millions(self):
        assert _format_cap(500e6) == "$500M"
