"""Two Sigma-style macro market outlook — 3-6 month horizon."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from takata_core.data.feed_market import MarketDataFeed
from takata_core.indicators.technical import ema, rsi, atr

logger = logging.getLogger(__name__)

# Macro proxy tickers
MACRO_TICKERS = {
    "equity": "SPY",
    "nasdaq": "QQQ",
    "small_cap": "IWM",
    "bonds_long": "TLT",
    "bonds_short": "SHY",
    "high_yield": "HYG",
    "investment_grade": "LQD",
    "gold": "GLD",
    "oil": "USO",
    "dollar": "UUP",
    "vix": "^VIX",
    "emerging": "EEM",
    "real_estate": "VNQ",
}


@dataclass
class MacroSignal:
    """A single macro indicator reading."""
    name: str
    value: str
    signal: str  # "bullish", "bearish", "neutral"
    description: str


class MacroOutlook:
    """Generates a 3-6 month macro market outlook.

    Analyzes cross-asset signals: equities, bonds, credit, commodities,
    volatility, and breadth to determine market stance.
    """

    def __init__(self, period: str = "1y"):
        self.period = period
        self._signals: List[MacroSignal] = []
        self._data: Dict[str, pd.DataFrame] = {}

    def _load(self, key: str) -> Optional[pd.DataFrame]:
        if key in self._data:
            return self._data[key]
        ticker = MACRO_TICKERS.get(key)
        if not ticker:
            return None
        try:
            feed = MarketDataFeed(ticker, period=self.period, interval="1d")
            df = feed.load()
            self._data[key] = df
            return df
        except Exception as e:
            logger.warning("Could not load %s (%s): %s", key, ticker, e)
            return None

    def _ret(self, key: str, bars: int) -> Optional[float]:
        df = self._load(key)
        if df is None or len(df) < bars:
            return None
        return float((df["close"].iloc[-1] / df["close"].iloc[-bars] - 1) * 100)

    def analyze(self) -> List[MacroSignal]:
        """Run full macro analysis."""
        self._signals = []

        # 1. Equity trend
        spy = self._load("equity")
        if spy is not None and len(spy) > 200:
            close = spy["close"]
            sma200 = close.rolling(200).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            price = close.iloc[-1]
            if price > sma200 and sma50 > sma200:
                sig = "bullish"
                desc = f"SPY above 200 SMA, golden cross intact"
            elif price < sma200 and sma50 < sma200:
                sig = "bearish"
                desc = f"SPY below 200 SMA, death cross"
            else:
                sig = "neutral"
                desc = f"SPY mixed signals vs moving averages"
            self._signals.append(MacroSignal("Equity Trend", f"${price:.2f}", sig, desc))

        # 2. Market breadth (SPY vs IWM — small caps leading/lagging)
        spy_ret = self._ret("equity", 63)
        iwm_ret = self._ret("small_cap", 63)
        if spy_ret is not None and iwm_ret is not None:
            spread = iwm_ret - spy_ret
            if spread > 3:
                sig = "bullish"
                desc = f"Small caps leading by {spread:+.1f}% (risk appetite)"
            elif spread < -3:
                sig = "bearish"
                desc = f"Small caps lagging by {spread:.1f}% (risk aversion)"
            else:
                sig = "neutral"
                desc = f"Small/large cap spread: {spread:+.1f}%"
            self._signals.append(MacroSignal("Market Breadth", f"{spread:+.1f}%", sig, desc))

        # 3. Credit markets (HYG vs LQD spread behavior)
        hyg_ret = self._ret("high_yield", 63)
        lqd_ret = self._ret("investment_grade", 63)
        if hyg_ret is not None and lqd_ret is not None:
            credit_spread = hyg_ret - lqd_ret
            if credit_spread > 2:
                sig = "bullish"
                desc = f"HY outperforming IG by {credit_spread:+.1f}% (credit appetite)"
            elif credit_spread < -2:
                sig = "bearish"
                desc = f"HY underperforming IG by {credit_spread:.1f}% (credit stress)"
            else:
                sig = "neutral"
                desc = f"Credit spread stable ({credit_spread:+.1f}%)"
            self._signals.append(MacroSignal("Credit Markets", f"{credit_spread:+.1f}%", sig, desc))

        # 4. Yield curve proxy (TLT vs SHY)
        tlt_ret = self._ret("bonds_long", 63)
        shy_ret = self._ret("bonds_short", 63)
        if tlt_ret is not None and shy_ret is not None:
            curve = tlt_ret - shy_ret
            if curve > 5:
                sig = "bearish"
                desc = f"Long bonds rallying ({curve:+.1f}%) — flight to safety / rate cut expectations"
            elif curve < -5:
                sig = "bullish"
                desc = f"Long bonds selling off ({curve:.1f}%) — growth expectations rising"
            else:
                sig = "neutral"
                desc = f"Yield curve proxy stable ({curve:+.1f}%)"
            self._signals.append(MacroSignal("Rate Expectations", f"{curve:+.1f}%", sig, desc))

        # 5. Gold / safe haven
        gold_ret = self._ret("gold", 63)
        if gold_ret is not None:
            if gold_ret > 8:
                sig = "bearish"
                desc = f"Gold rallying {gold_ret:+.1f}% — safe haven demand elevated"
            elif gold_ret < -3:
                sig = "bullish"
                desc = f"Gold declining {gold_ret:.1f}% — risk appetite returning"
            else:
                sig = "neutral"
                desc = f"Gold return: {gold_ret:+.1f}%"
            self._signals.append(MacroSignal("Safe Haven (Gold)", f"{gold_ret:+.1f}%", sig, desc))

        # 6. USD strength
        usd_ret = self._ret("dollar", 63)
        if usd_ret is not None:
            if usd_ret > 3:
                sig = "bearish"
                desc = f"Strong dollar ({usd_ret:+.1f}%) — headwind for earnings and EM"
            elif usd_ret < -3:
                sig = "bullish"
                desc = f"Weak dollar ({usd_ret:.1f}%) — tailwind for multinationals and EM"
            else:
                sig = "neutral"
                desc = f"USD stable ({usd_ret:+.1f}%)"
            self._signals.append(MacroSignal("USD Strength", f"{usd_ret:+.1f}%", sig, desc))

        # 7. VIX level
        vix = self._load("vix")
        if vix is not None:
            vix_val = float(vix["close"].iloc[-1])
            if vix_val > 25:
                sig = "bearish"
                desc = f"VIX at {vix_val:.1f} — elevated fear, expect volatility"
            elif vix_val < 15:
                sig = "bullish"
                desc = f"VIX at {vix_val:.1f} — low fear, complacency risk"
            else:
                sig = "neutral"
                desc = f"VIX at {vix_val:.1f} — moderate uncertainty"
            self._signals.append(MacroSignal("Volatility (VIX)", f"{vix_val:.1f}", sig, desc))

        # 8. Emerging markets
        em_ret = self._ret("emerging", 63)
        if em_ret is not None:
            if em_ret > 5:
                sig = "bullish"
                desc = f"EM rallying {em_ret:+.1f}% — global growth broadening"
            elif em_ret < -5:
                sig = "bearish"
                desc = f"EM declining {em_ret:.1f}% — global growth concerns"
            else:
                sig = "neutral"
                desc = f"EM return: {em_ret:+.1f}%"
            self._signals.append(MacroSignal("Emerging Markets", f"{em_ret:+.1f}%", sig, desc))

        return self._signals

    def market_stance(self) -> str:
        """Overall market stance based on signal balance."""
        if not self._signals:
            self.analyze()
        bull = sum(1 for s in self._signals if s.signal == "bullish")
        bear = sum(1 for s in self._signals if s.signal == "bearish")
        total = len(self._signals)
        if bull > bear + 2:
            return "BULLISH"
        elif bear > bull + 2:
            return "BEARISH"
        elif bull > bear:
            return "CAUTIOUSLY BULLISH"
        elif bear > bull:
            return "CAUTIOUSLY BEARISH"
        return "NEUTRAL"

    def format_report(self) -> str:
        if not self._signals:
            self.analyze()

        w = 75
        stance = self.market_stance()
        bull = sum(1 for s in self._signals if s.signal == "bullish")
        bear = sum(1 for s in self._signals if s.signal == "bearish")
        neutral = sum(1 for s in self._signals if s.signal == "neutral")

        lines = [
            "=" * w,
            "  MACRO MARKET OUTLOOK — TWO SIGMA STRATEGY",
            f"  Horizon: 3-6 months  |  Stance: {stance}",
            f"  Signal Balance: {bull} bullish / {neutral} neutral / {bear} bearish",
            "=" * w,
            "",
            "MACRO INDICATORS",
            "-" * w,
        ]

        for s in self._signals:
            icon = {"bullish": "+", "bearish": "-", "neutral": "~"}[s.signal]
            lines.append(f"  [{icon}] {s.name:25s}  {s.value:>10s}  {s.signal.upper():10s}")
            lines.append(f"      {s.description}")
            lines.append("")

        # Tactical recommendations
        lines.extend(["TACTICAL POSITIONING", "-" * w])

        if stance in ("BULLISH", "CAUTIOUSLY BULLISH"):
            lines.extend([
                "  Equities:     Overweight (favor growth / cyclicals)",
                "  Bonds:        Underweight (short duration preferred)",
                "  Commodities:  Neutral to overweight",
                "  Cash:         Minimal",
            ])
        elif stance in ("BEARISH", "CAUTIOUSLY BEARISH"):
            lines.extend([
                "  Equities:     Underweight (favor defensives / quality)",
                "  Bonds:        Overweight (extend duration for safety)",
                "  Commodities:  Underweight (except gold)",
                "  Cash:         Elevated (5-15%)",
            ])
        else:
            lines.extend([
                "  Equities:     Equal weight (balanced sector exposure)",
                "  Bonds:        Equal weight (neutral duration)",
                "  Commodities:  Equal weight",
                "  Cash:         Moderate (3-5%)",
            ])

        # Risk hedges
        lines.extend(["", "RISK HEDGES", "-" * w])
        if bear >= 3:
            lines.append("  >> Multiple bearish signals — consider SPY puts or VIX calls")
            lines.append("  >> Reduce gross exposure and tighten stops")
        elif bear >= 2:
            lines.append("  >> Some caution warranted — consider collar strategies")
        else:
            lines.append("  >> Low hedging urgency — maintain standard risk budget")

        lines.extend(["", "=" * w])
        return "\n".join(lines)
