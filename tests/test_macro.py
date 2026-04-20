"""Unit tests for macro outlook module."""

import pytest

from takata_engine.macro.outlook import MacroOutlook


class TestMacroOutlook:
    def test_analyze_returns_signals(self):
        outlook = MacroOutlook(period="6mo")
        signals = outlook.analyze()
        assert len(signals) > 0
        for s in signals:
            assert s.signal in ("bullish", "bearish", "neutral")
            assert s.name
            assert s.description

    def test_market_stance(self):
        outlook = MacroOutlook(period="6mo")
        outlook.analyze()
        stance = outlook.market_stance()
        assert stance in (
            "BULLISH", "CAUTIOUSLY BULLISH", "NEUTRAL",
            "CAUTIOUSLY BEARISH", "BEARISH"
        )
