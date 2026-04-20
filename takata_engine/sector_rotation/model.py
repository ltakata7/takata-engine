"""Citadel-style sector rotation model — ETF allocation based on momentum, valuation, macro."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from takata_engine.data.feed_market import MarketDataFeed
from takata_engine.indicators.technical import ema, rsi

logger = logging.getLogger(__name__)

SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Disc.": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Communication": "XLC",
}

RISK_ON_SECTORS = {"Technology", "Consumer Disc.", "Financials", "Industrials"}
RISK_OFF_SECTORS = {"Utilities", "Consumer Staples", "Healthcare", "Real Estate"}


@dataclass
class SectorScore:
    """Scoring result for a single sector."""
    sector: str
    etf: str
    momentum_score: float  # 0-100
    relative_strength: float  # vs SPY
    rsi_value: float
    return_3m: float
    return_6m: float
    composite: float
    recommendation: str  # "overweight", "equal", "underweight"
    allocation_pct: float


class SectorRotationModel:
    """Sector allocation model using momentum and relative strength.

    Parameters
    ----------
    period : str
        Lookback period for data.
    """

    def __init__(self, period: str = "1y"):
        self.period = period
        self._scores: Optional[List[SectorScore]] = None

    def analyze(self) -> List[SectorScore]:
        """Score all sectors and generate allocations."""
        # Fetch SPY as benchmark
        spy_feed = MarketDataFeed("SPY", period=self.period, interval="1d")
        spy = spy_feed.load()
        spy_close = spy["close"]

        scores = []
        for sector, etf in SECTOR_ETFS.items():
            try:
                feed = MarketDataFeed(etf, period=self.period, interval="1d")
                df = feed.load()
                close = df["close"]

                # Momentum: 3M and 6M returns
                ret_3m = float((close.iloc[-1] / close.iloc[-63] - 1) * 100) if len(close) >= 63 else 0
                ret_6m = float((close.iloc[-1] / close.iloc[-126] - 1) * 100) if len(close) >= 126 else 0

                # Relative strength vs SPY
                if len(close) >= 63 and len(spy_close) >= 63:
                    rel = (close / spy_close.reindex(close.index, method="ffill")).dropna()
                    rs = float((rel.iloc[-1] / rel.iloc[-63] - 1) * 100) if len(rel) >= 63 else 0
                else:
                    rs = 0

                # RSI
                rsi_val = float(rsi(close, 14).iloc[-1])

                # Composite momentum score (0-100)
                mom_score = max(0, min(100, 50 + ret_3m * 1.5 + ret_6m * 0.5 + rs * 2))

                scores.append(SectorScore(
                    sector=sector,
                    etf=etf,
                    momentum_score=round(mom_score, 1),
                    relative_strength=round(rs, 2),
                    rsi_value=round(rsi_val, 1),
                    return_3m=round(ret_3m, 2),
                    return_6m=round(ret_6m, 2),
                    composite=round(mom_score, 1),
                    recommendation="",
                    allocation_pct=0,
                ))
            except Exception as e:
                logger.warning("Failed to analyze %s (%s): %s", sector, etf, e)

        # Rank and assign recommendations
        scores.sort(key=lambda s: -s.composite)
        n = len(scores)
        top_third = n // 3
        bottom_third = n - top_third

        total_alloc = 0
        for i, s in enumerate(scores):
            if i < top_third:
                s.recommendation = "OVERWEIGHT"
                s.allocation_pct = round(15.0 / top_third, 1) if top_third > 0 else 15.0
            elif i >= bottom_third:
                s.recommendation = "UNDERWEIGHT"
                s.allocation_pct = round(5.0 / (n - bottom_third), 1) if (n - bottom_third) > 0 else 5.0
            else:
                s.recommendation = "EQUAL WEIGHT"
                mid_count = bottom_third - top_third
                s.allocation_pct = round((100 - 15 * 2) / max(mid_count, 1) * (mid_count / n), 1)

        # Normalize to 100%
        total = sum(s.allocation_pct for s in scores)
        if total > 0:
            for s in scores:
                s.allocation_pct = round(s.allocation_pct / total * 100, 1)

        self._scores = scores
        return scores

    def _risk_stance(self) -> str:
        """Determine risk-on vs risk-off based on sector leadership."""
        if not self._scores:
            return "NEUTRAL"
        top3 = {s.sector for s in self._scores[:3]}
        risk_on_count = len(top3 & RISK_ON_SECTORS)
        risk_off_count = len(top3 & RISK_OFF_SECTORS)
        if risk_on_count >= 2:
            return "RISK-ON"
        elif risk_off_count >= 2:
            return "RISK-OFF"
        return "NEUTRAL"

    def format_report(self) -> str:
        """Format as Citadel-style sector allocation note."""
        if self._scores is None:
            self.analyze()

        w = 80
        lines = [
            "=" * w,
            "  SECTOR ROTATION MODEL — CITADEL MACRO STRATEGY",
            f"  Market Stance: {self._risk_stance()}",
            "=" * w,
            "",
            "SECTOR RANKINGS",
            "-" * w,
            f"  {'Rank':>4s}  {'Sector':18s} {'ETF':5s} {'Score':>6s} "
            f"{'3M Ret':>7s} {'6M Ret':>7s} {'RS':>7s} {'RSI':>5s} {'Alloc':>6s}  Rec",
        ]

        for i, s in enumerate(self._scores):
            lines.append(
                f"  {i+1:>4d}  {s.sector:18s} {s.etf:5s} {s.composite:>5.0f}  "
                f"{s.return_3m:>+6.1f}% {s.return_6m:>+6.1f}% {s.relative_strength:>+6.1f}% "
                f"{s.rsi_value:>5.0f} {s.allocation_pct:>5.1f}%  {s.recommendation}"
            )

        # Overweight / Underweight summary
        ow = [s for s in self._scores if s.recommendation == "OVERWEIGHT"]
        uw = [s for s in self._scores if s.recommendation == "UNDERWEIGHT"]

        lines.extend(["", "ALLOCATION SUMMARY", "-" * w])
        lines.append("  OVERWEIGHT:")
        for s in ow:
            lines.append(f"    {s.etf} ({s.sector}) — {s.allocation_pct:.1f}%  "
                         f"[3M: {s.return_3m:+.1f}%, RS: {s.relative_strength:+.1f}%]")

        lines.append("  UNDERWEIGHT:")
        for s in uw:
            lines.append(f"    {s.etf} ({s.sector}) — {s.allocation_pct:.1f}%  "
                         f"[3M: {s.return_3m:+.1f}%, RS: {s.relative_strength:+.1f}%]")

        lines.extend(["", "=" * w])
        return "\n".join(lines)
