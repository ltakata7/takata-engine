"""Goldman Sachs-style equity research report generator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from takata_core.research.fundamentals import (
    _format_big,
    get_fundamentals,
    moat_score,
    peer_comparison,
)

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Complete equity research analysis."""

    fundamentals: Dict[str, Any]
    moat: Dict[str, Any]
    peers: pd.DataFrame
    bull_case: Dict[str, Any]
    bear_case: Dict[str, Any]
    conviction: str


class EquityResearchReport:
    """Generates institutional-grade equity research notes.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    peers : list of str, optional
        Peer tickers for comparison. Auto-detected if not provided.
    """

    def __init__(self, ticker: str, peers: Optional[List[str]] = None):
        self.ticker = ticker.upper()
        self.peers = peers
        self._result: Optional[ResearchResult] = None

    def generate(self) -> ResearchResult:
        """Run full analysis."""
        f = get_fundamentals(self.ticker)
        moat = moat_score(f)
        peers_df = peer_comparison(self.ticker, self.peers)

        # Bull/bear cases derived from fundamentals
        bull = self._build_bull_case(f)
        bear = self._build_bear_case(f)
        conviction = self._compute_conviction(f, moat)

        self._result = ResearchResult(
            fundamentals=f,
            moat=moat,
            peers=peers_df,
            bull_case=bull,
            bear_case=bear,
            conviction=conviction,
        )
        return self._result

    def _build_bull_case(self, f: Dict) -> Dict[str, Any]:
        drivers = []
        price = f.get("price", 0)

        rg = f.get("revenue_growth")
        if rg and rg > 0.10:
            drivers.append(f"Strong revenue growth ({rg:.0%}) with runway for expansion")
        pm = f.get("profit_margin")
        if pm and pm > 0.15:
            drivers.append(f"Best-in-class margins ({pm:.0%}) indicating pricing power")
        fcf = f.get("free_cf", 0)
        if fcf > 0:
            drivers.append(f"Healthy FCF generation ({_format_big(fcf)}) supports buybacks/dividends")
        roe = f.get("roe")
        if roe and roe > 0.15:
            drivers.append(f"Superior capital efficiency (ROE {roe:.0%})")
        target = f.get("target_high")
        if target and target > price:
            drivers.append(f"Street high target at ${target:.2f} ({(target/price-1)*100:+.0f}%)")

        if not drivers:
            drivers.append("Limited visible catalysts at current levels")

        # Bull price target: analyst high or 15% above current
        bull_target = f.get("target_high") or price * 1.15

        return {"drivers": drivers, "price_target": round(bull_target, 2)}

    def _build_bear_case(self, f: Dict) -> Dict[str, Any]:
        risks = []
        price = f.get("price", 0)

        pe = f.get("pe_trailing")
        if pe and pe > 30:
            risks.append(f"Elevated valuation (PE {pe:.0f}x) leaves little margin of safety")
        de = f.get("debt_to_equity")
        if de and de > 100:
            risks.append(f"Leveraged balance sheet (D/E {de:.0f}) increases downside risk")
        eg = f.get("earnings_growth")
        if eg and eg < 0:
            risks.append(f"Declining earnings ({eg:.0%}) signals deteriorating fundamentals")
        rg = f.get("revenue_growth")
        if rg and rg < 0:
            risks.append(f"Revenue contraction ({rg:.0%}) suggests market share loss")
        vol = f.get("volatility_1y")
        if vol and vol > 35:
            risks.append(f"High volatility ({vol:.0f}%) increases drawdown risk")

        if not risks:
            risks.append("No major red flags identified at current levels")

        # Bear price target: analyst low or 15% below current
        bear_target = f.get("target_low") or price * 0.85

        return {"risks": risks, "price_target": round(bear_target, 2)}

    def _compute_conviction(self, f: Dict, moat: Dict) -> str:
        """Compute conviction rating: Conviction Buy / Buy / Neutral / Sell / Conviction Sell."""
        score = 0

        # Moat score contribution
        moat_total = moat.get("total", 5)
        score += (moat_total - 5) * 2  # -10 to +10

        # Valuation vs target
        price = f.get("price", 0)
        target = f.get("target_mean")
        if target and price > 0:
            upside = (target / price - 1) * 100
            score += min(10, max(-10, upside * 0.5))

        # Growth
        rg = f.get("revenue_growth") or 0
        score += min(5, max(-5, rg * 20))

        # Analyst consensus
        rec = f.get("recommendation_mean")
        if rec:
            score += (3 - rec) * 3  # 1=strong buy→+6, 5=sell→-6

        if score > 8:
            return "CONVICTION BUY"
        elif score > 3:
            return "BUY"
        elif score > -3:
            return "NEUTRAL"
        elif score > -8:
            return "SELL"
        else:
            return "CONVICTION SELL"

    def format_report(self) -> str:
        """Format as Goldman Sachs-style equity research note."""
        r = self._result
        if r is None:
            r = self.generate()

        f = r.fundamentals
        w = 70
        price = f["price"]

        lines = [
            "=" * w,
            f"  EQUITY RESEARCH NOTE — {f['ticker']}",
            f"  {f['name']}  |  {f['sector']} / {f['industry']}",
            f"  Rating: {r.conviction}",
            "=" * w,
            "",
            f"  Price: ${price:.2f}  |  Mkt Cap: {f['market_cap_label']}  |  "
            f"EV: {_format_big(f['enterprise_value'])}",
        ]
        if f.get("price_52w_high"):
            lines.append(
                f"  52w Range: ${f['price_52w_low']:.2f} - ${f['price_52w_high']:.2f}  "
                f"({f['price_vs_52w_high']:+.1f}% from high)"
            )

        # Business model
        desc = f.get("description", "")
        if desc:
            # Truncate to ~200 chars
            short_desc = desc[:200] + "..." if len(desc) > 200 else desc
            lines.extend(["", "BUSINESS MODEL", "-" * w, f"  {short_desc}"])

        # Revenue & profitability
        lines.extend([
            "", "REVENUE & PROFITABILITY", "-" * w,
            f"  Revenue:          {_format_big(f['revenue'])}  "
            f"(growth: {f.get('revenue_growth', 0) or 0:.0%})",
            f"  EBITDA:           {_format_big(f['ebitda'])}",
            f"  Net Income:       {_format_big(f['net_income'])}",
            f"  EPS (TTM/Fwd):    ${f.get('eps_trailing', 0) or 0:.2f} / "
            f"${f.get('eps_forward', 0) or 0:.2f}",
            "",
            f"  Gross Margin:     {(f.get('gross_margin') or 0):.1%}",
            f"  Operating Margin: {(f.get('operating_margin') or 0):.1%}",
            f"  Net Margin:       {(f.get('profit_margin') or 0):.1%}",
            f"  ROE:              {(f.get('roe') or 0):.1%}",
            f"  ROA:              {(f.get('roa') or 0):.1%}",
        ])

        # Balance sheet
        lines.extend([
            "", "BALANCE SHEET", "-" * w,
            f"  Cash:             {_format_big(f['total_cash'])}",
            f"  Debt:             {_format_big(f['total_debt'])}",
            f"  Net Debt:         {_format_big(f['total_debt'] - f['total_cash'])}",
            f"  D/E Ratio:        {f.get('debt_to_equity', 0) or 0:.0f}",
            f"  Current Ratio:    {f.get('current_ratio', 0) or 0:.2f}",
        ])

        # Free cash flow
        lines.extend([
            "", "FREE CASH FLOW", "-" * w,
            f"  Operating CF:     {_format_big(f['operating_cf'])}",
            f"  Free Cash Flow:   {_format_big(f['free_cf'])}",
            f"  FCF Yield:        {f.get('fcf_yield', 0):.1f}%",
        ])

        # Moat scoring
        moat = r.moat
        lines.extend([
            "", f"COMPETITIVE MOAT SCORE: {moat['total']}/10", "-" * w,
        ])
        for factor, score in moat["breakdown"].items():
            bar = "#" * int(score * 5)
            lines.append(f"  {factor.capitalize():15s} {score:.1f}/2.0  {bar}")

        # Valuation
        lines.extend([
            "", "VALUATION", "-" * w,
            f"  P/E (TTM):        {f.get('pe_trailing') or 'N/A'}",
            f"  P/E (Fwd):        {f.get('pe_forward') or 'N/A'}",
            f"  PEG:              {f.get('peg_ratio') or 'N/A'}",
            f"  P/B:              {f.get('pb_ratio') or 'N/A'}",
            f"  P/S:              {f.get('ps_ratio') or 'N/A'}",
            f"  EV/EBITDA:        {f.get('ev_ebitda') or 'N/A'}",
        ])

        # Peer comparison
        if not r.peers.empty:
            lines.extend(["", "PEER COMPARISON", "-" * w])
            peer_str = r.peers.to_string(index=False, float_format="%.1f")
            for line in peer_str.split("\n"):
                lines.append(f"  {line}")

        # Bull case
        lines.extend([
            "", "BULL CASE — Price Target: ${:.2f}".format(r.bull_case["price_target"]),
            "-" * w,
        ])
        for d in r.bull_case["drivers"]:
            lines.append(f"  + {d}")

        # Bear case
        lines.extend([
            "", "BEAR CASE — Price Target: ${:.2f}".format(r.bear_case["price_target"]),
            "-" * w,
        ])
        for risk in r.bear_case["risks"]:
            lines.append(f"  - {risk}")

        # Conviction
        lines.extend([
            "",
            "=" * w,
            f"  CONVICTION RATING:  {r.conviction}",
            f"  Bull Target: ${r.bull_case['price_target']:.2f}  |  "
            f"Bear Target: ${r.bear_case['price_target']:.2f}  |  "
            f"Analyst Mean: ${f.get('target_mean') or 0:.2f}",
            "=" * w,
        ])

        return "\n".join(lines)
