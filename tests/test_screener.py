"""Unit tests for multi-factor screener."""

import numpy as np
import pytest

from takata_core.screener.factors import (
    compute_all_factors,
    growth_score,
    momentum_score,
    quality_score,
    sentiment_score,
    value_score,
)


class TestValueScore:
    def test_low_pe_scores_high(self):
        info = {"trailingPE": 10, "priceToBook": 1.0, "dividendYield": 0.03, "enterpriseToEbitda": 8}
        result = value_score(info)
        assert result["score"] > 60

    def test_high_pe_scores_low(self):
        info = {"trailingPE": 50, "priceToBook": 10.0, "dividendYield": 0.0, "enterpriseToEbitda": 30}
        result = value_score(info)
        assert result["score"] < 40

    def test_empty_info(self):
        result = value_score({})
        assert result["score"] == 50.0  # default


class TestQualityScore:
    def test_high_quality(self):
        info = {"returnOnEquity": 0.25, "profitMargins": 0.20, "debtToEquity": 30, "currentRatio": 2.0}
        result = quality_score(info)
        assert result["score"] > 60

    def test_low_quality(self):
        info = {"returnOnEquity": 0.02, "profitMargins": 0.02, "debtToEquity": 300, "currentRatio": 0.5}
        result = quality_score(info)
        assert result["score"] < 40


class TestMomentumScore:
    def test_returns_dict_structure(self):
        # Use a known ticker that should work
        result = momentum_score("SPY")
        assert "score" in result
        assert "details" in result
        assert 0 <= result["score"] <= 100


class TestGrowthScore:
    def test_high_growth(self):
        info = {"revenueGrowth": 0.30, "earningsGrowth": 0.40, "forwardPE": 15, "trailingPE": 25}
        result = growth_score(info)
        assert result["score"] > 60

    def test_negative_growth(self):
        info = {"revenueGrowth": -0.20, "earningsGrowth": -0.30}
        result = growth_score(info)
        assert result["score"] < 50


class TestComposite:
    def test_composite_in_range(self):
        result = compute_all_factors("AAPL")
        assert 0 <= result["composite"] <= 100
        assert result["ticker"] == "AAPL"
        for factor in ["value", "quality", "momentum", "growth", "sentiment"]:
            assert factor in result
            assert 0 <= result[factor]["score"] <= 100
