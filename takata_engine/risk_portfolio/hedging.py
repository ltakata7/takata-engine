"""Hedging recommendations based on portfolio risk profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class HedgeRecommendation:
    """A single hedging recommendation."""

    instrument: str
    action: str  # "buy", "sell", "consider"
    rationale: str
    priority: str  # "high", "medium", "low"


def recommend_hedges(
    risk_metrics: Dict[str, Any],
    sector_weights: Dict[str, float],
    asset_class_weights: Dict[str, float],
    beta: float,
    portfolio_value: float,
) -> List[HedgeRecommendation]:
    """Generate hedging recommendations based on risk analysis.

    Parameters
    ----------
    risk_metrics : dict
        Output from ``compute_risk_metrics``.
    sector_weights : dict
        Sector → weight mapping.
    asset_class_weights : dict
        Asset class → weight mapping.
    beta : float
        Portfolio beta vs benchmark.
    portfolio_value : float
        Total portfolio value.
    """
    recommendations = []

    # 1. High beta → consider SPY puts or VIX calls
    if beta > 1.2:
        recommendations.append(HedgeRecommendation(
            instrument="SPY puts / UVXY",
            action="consider",
            rationale=f"Portfolio beta is {beta:.2f} — significantly above market. "
                      f"Consider protective puts on SPY or small VIX hedge.",
            priority="high",
        ))

    # 2. High volatility → reduce position sizes or add bonds
    vol = risk_metrics.get("annualized_volatility", 0)
    if vol > 0.20:
        recommendations.append(HedgeRecommendation(
            instrument="TLT / AGG / BND",
            action="consider",
            rationale=f"Annualized volatility is {vol:.1%} — above 20% threshold. "
                      f"Consider increasing bond allocation to reduce portfolio vol.",
            priority="high" if vol > 0.25 else "medium",
        ))

    # 3. Tech concentration > 30%
    tech_weight = sector_weights.get("Technology", 0) + sector_weights.get("US Equity", 0) * 0.3
    actual_tech = sector_weights.get("Technology", 0)
    if actual_tech > 0.30:
        recommendations.append(HedgeRecommendation(
            instrument="QQQ puts / XLK puts",
            action="consider",
            rationale=f"Technology sector is {actual_tech:.1%} of portfolio — "
                      f"concentrated. Consider sector hedges or diversification.",
            priority="medium",
        ))

    # 4. No bond allocation
    bond_weight = asset_class_weights.get("US Bonds", 0)
    if bond_weight < 0.05:
        recommendations.append(HedgeRecommendation(
            instrument="AGG / BND / TLT",
            action="buy",
            rationale="No meaningful bond allocation. Bonds provide crisis diversification "
                      "and reduce portfolio drawdowns.",
            priority="medium",
        ))

    # 5. No commodities/gold allocation
    commodity_weight = asset_class_weights.get("Commodities", 0)
    if commodity_weight < 0.03:
        recommendations.append(HedgeRecommendation(
            instrument="GLD / IAU",
            action="consider",
            rationale="No gold/commodities allocation. Gold provides inflation hedge "
                      "and tail risk protection.",
            priority="low",
        ))

    # 6. High VaR
    var95 = risk_metrics.get("var_95", 0)
    if var95 > 0.025:
        daily_var_usd = var95 * portfolio_value
        recommendations.append(HedgeRecommendation(
            instrument="Portfolio-level tail hedge",
            action="consider",
            rationale=f"Daily 95% VaR is {var95:.2%} (${daily_var_usd:,.0f}). "
                      f"Consider tail-risk hedging via OTM puts or put spreads.",
            priority="high" if var95 > 0.035 else "medium",
        ))

    # 7. High max drawdown
    dd_info = risk_metrics.get("max_drawdown", {})
    max_dd = dd_info.get("max_drawdown", 0) if isinstance(dd_info, dict) else 0
    if max_dd > 0.15:
        recommendations.append(HedgeRecommendation(
            instrument="Systematic rebalancing",
            action="consider",
            rationale=f"Historical max drawdown is {max_dd:.1%}. Consider systematic "
                      f"rebalancing triggers and/or trailing stop strategy.",
            priority="medium",
        ))

    if not recommendations:
        recommendations.append(HedgeRecommendation(
            instrument="No action needed",
            action="hold",
            rationale="Portfolio risk metrics are within acceptable bounds.",
            priority="low",
        ))

    return recommendations
