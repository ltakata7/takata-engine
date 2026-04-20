"""Risk dashboard — Bridgewater-style portfolio risk report."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from takata_engine.data.feed_market import MarketDataFeed
from takata_engine.risk_portfolio.analytics import (
    compute_risk_metrics,
    correlation_matrix,
    rolling_beta,
)
from takata_engine.risk_portfolio.concentration import (
    asset_class_breakdown,
    herfindahl_index,
    sector_breakdown,
    top_holdings_concentration,
)
from takata_engine.risk_portfolio.hedging import HedgeRecommendation, recommend_hedges
from takata_engine.risk_portfolio.portfolio import Portfolio
from takata_engine.risk_portfolio.stress import ScenarioResult, run_all_scenarios

logger = logging.getLogger(__name__)


@dataclass
class RiskResult:
    """Complete risk analysis result."""

    portfolio: Portfolio
    risk_metrics: Dict[str, Any]
    sector_weights: Dict[str, float]
    asset_class_weights: Dict[str, float]
    concentration_hhi: float
    top5_concentration: float
    correlation: pd.DataFrame
    stress_results: List[ScenarioResult]
    hedge_recommendations: List[HedgeRecommendation]


class RiskDashboard:
    """Generates a Bridgewater-style portfolio risk dashboard.

    Parameters
    ----------
    portfolio_path : str
        Path to portfolio YAML (or account name like ``"takata"``).
    period : str
        Historical lookback period for analytics.
    """

    def __init__(self, portfolio_path: str, period: str = "1y"):
        self.portfolio_path = portfolio_path
        self.period = period
        self._result: Optional[RiskResult] = None

    def generate(self) -> RiskResult:
        """Run full risk analysis."""
        portfolio = Portfolio.load(self.portfolio_path)
        logger.info("Loaded portfolio '%s': %d holdings, $%.0f",
                     portfolio.name, len(portfolio.holdings), portfolio.total_value)

        # Portfolio returns
        port_returns = portfolio.portfolio_returns(self.period)

        # Benchmark returns
        bench_feed = MarketDataFeed(portfolio.benchmark, period=self.period, interval="1d")
        bench_df = bench_feed.load()
        bench_returns = bench_df["close"].pct_change().dropna()
        bench_returns.name = portfolio.benchmark

        # Risk metrics
        risk_metrics = compute_risk_metrics(port_returns, bench_returns)

        # Correlation
        returns_df = portfolio.returns_history(self.period)
        corr = correlation_matrix(returns_df) if len(returns_df) > 10 else pd.DataFrame()

        # Concentration
        sectors = sector_breakdown(portfolio.holdings)
        asset_classes = asset_class_breakdown(portfolio.holdings)
        hhi = herfindahl_index(portfolio.holdings)
        top5 = top_holdings_concentration(portfolio.holdings, 5)

        # Stress tests
        stress = run_all_scenarios(portfolio.holdings, portfolio.total_value)

        # Hedging
        hedges = recommend_hedges(
            risk_metrics=risk_metrics,
            sector_weights=sectors,
            asset_class_weights=asset_classes,
            beta=risk_metrics["beta"],
            portfolio_value=portfolio.total_value,
        )

        self._result = RiskResult(
            portfolio=portfolio,
            risk_metrics=risk_metrics,
            sector_weights=sectors,
            asset_class_weights=asset_classes,
            concentration_hhi=hhi,
            top5_concentration=top5,
            correlation=corr,
            stress_results=stress,
            hedge_recommendations=hedges,
        )
        return self._result

    def format_report(self) -> str:
        """Format as Bridgewater-style risk dashboard."""
        r = self._result
        if r is None:
            r = self.generate()

        p = r.portfolio
        m = r.risk_metrics
        w = 65

        lines = [
            "=" * w,
            f"  PORTFOLIO RISK ASSESSMENT — {p.name.upper()}",
            f"  Account: {p.account_type}  |  Benchmark: {p.benchmark}",
            "=" * w,
            "",
            "PORTFOLIO OVERVIEW",
            "-" * w,
        ]

        # Holdings table
        hdf = p.holdings_df()
        lines.append(f"  {'Ticker':8s} {'Shares':>8s} {'Price':>10s} {'Value':>12s} {'Weight':>8s} {'G/L%':>8s}")
        for _, row in hdf.iterrows():
            lines.append(
                f"  {row['ticker']:8s} {row['shares']:>8.0f} "
                f"${row['current_price']:>9.2f} ${row['market_value']:>11,.0f} "
                f"{row['weight']:>7.1%} {row['gain_loss_pct']:>+7.1f}%"
            )
        lines.append(f"  {'':8s} {'':>8s} {'Total':>10s} ${p.total_value:>11,.0f}")
        lines.append(f"  {'':8s} {'':>8s} {'P&L':>10s} ${p.total_gain_loss:>11,.0f}")

        # Risk metrics
        dd = m["max_drawdown"]
        lines.extend([
            "",
            "RISK METRICS",
            "-" * w,
            f"  Annualized Volatility:    {m['annualized_volatility']:.1%}",
            f"  Beta vs {p.benchmark}:             {m['beta']:.2f}",
            f"  Sharpe Ratio:             {m['sharpe_ratio']:.2f}",
            f"  Annualized Return:        {m['annualized_return']:.1%}",
            f"  Daily VaR (95%):          {m['var_95']:.2%}  (${m['var_95'] * p.total_value:,.0f})",
            f"  Daily CVaR (95%):         {m['cvar_95']:.2%}  (${m['cvar_95'] * p.total_value:,.0f})",
            f"  Max Drawdown:             {dd['max_drawdown']:.1%}",
            f"    Peak:                   {dd['peak_date']}",
            f"    Trough:                 {dd['trough_date']}",
            f"    Recovery:               {dd['recovery_date'] or 'NOT RECOVERED'}",
        ])

        # Concentration
        lines.extend([
            "",
            "CONCENTRATION ANALYSIS",
            "-" * w,
            f"  Herfindahl Index (HHI):   {r.concentration_hhi:.3f}  "
            f"({'concentrated' if r.concentration_hhi > 0.15 else 'diversified'})",
            f"  Top 5 Holdings:           {r.top5_concentration:.1%}",
            "",
            "  Asset Class Allocation:",
        ])
        for ac, wt in r.asset_class_weights.items():
            bar = "#" * int(wt * 40)
            lines.append(f"    {ac:20s} {wt:>6.1%}  {bar}")

        lines.append("")
        lines.append("  Sector Allocation:")
        for sec, wt in r.sector_weights.items():
            bar = "#" * int(wt * 40)
            lines.append(f"    {sec:25s} {wt:>6.1%}  {bar}")

        # Correlation (top correlations)
        if not r.correlation.empty:
            lines.extend(["", "CORRELATION MATRIX (highlights)", "-" * w])
            corr = r.correlation
            pairs = []
            for i in range(len(corr)):
                for j in range(i + 1, len(corr)):
                    pairs.append((corr.index[i], corr.columns[j], corr.iloc[i, j]))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for t1, t2, c in pairs[:5]:
                label = "HIGH" if abs(c) > 0.7 else "MOD" if abs(c) > 0.4 else "LOW"
                lines.append(f"  {t1:6s} / {t2:6s}:  {c:+.2f}  ({label})")

        # Stress tests
        lines.extend(["", "STRESS TESTING", "-" * w])
        for s in r.stress_results:
            emoji = "!!!" if s.portfolio_impact_pct < -0.15 else "! " if s.portfolio_impact_pct < -0.10 else "  "
            lines.append(
                f"  {emoji} {s.name:25s}  {s.portfolio_impact_pct:>+7.1%}  "
                f"(${s.portfolio_impact_usd:>+12,.0f})"
            )
            if s.worst_holdings:
                worst = s.worst_holdings[0]
                lines.append(f"      Worst: {worst['ticker']} ({worst['shock_pct']:+.0%})")

        # Hedging
        lines.extend(["", "HEDGING RECOMMENDATIONS", "-" * w])
        for h in r.hedge_recommendations:
            priority_tag = f"[{h.priority.upper()}]"
            lines.append(f"  {priority_tag:8s} {h.instrument}")
            lines.append(f"           {h.rationale}")
            lines.append("")

        lines.append("=" * w)
        return "\n".join(lines)
