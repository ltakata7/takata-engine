"""Stress testing — predefined market scenarios applied to portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from takata_core.risk_portfolio.concentration import get_sector, get_asset_class
from takata_core.risk_portfolio.portfolio import Holding


@dataclass
class ScenarioResult:
    """Result of applying a stress scenario."""

    name: str
    description: str
    portfolio_impact_pct: float
    portfolio_impact_usd: float
    worst_holdings: List[Dict[str, Any]]
    best_holdings: List[Dict[str, Any]]


# ── Scenario definitions ──

SCENARIOS = {
    "recession": {
        "name": "Recession",
        "description": "Broad equity selloff, flight to quality",
        "shocks": {
            "US Equity": -0.20,
            "Intl Equity": -0.25,
            "US Bonds": 0.05,
            "REIT": -0.15,
            "Commodities": -0.10,
            # Sector-level overrides
            "Technology": -0.25,
            "Consumer Cyclical": -0.25,
            "Financial Services": -0.20,
            "Healthcare": -0.10,
            "Consumer Defensive": -0.08,
            "Utilities": -0.05,
        },
    },
    "rate_shock": {
        "name": "Rate Shock (+200bp)",
        "description": "Rapid rate increase, bond selloff, growth rotation",
        "shocks": {
            "US Equity": -0.08,
            "Intl Equity": -0.10,
            "US Bonds": -0.12,
            "REIT": -0.15,
            "Commodities": 0.05,
            "Technology": -0.15,
            "Utilities": -0.10,
            "Financial Services": 0.05,
        },
    },
    "market_crash": {
        "name": "Market Crash (-30%)",
        "description": "Severe equity crash, correlation spike",
        "shocks": {
            "US Equity": -0.30,
            "Intl Equity": -0.35,
            "US Bonds": 0.08,
            "REIT": -0.25,
            "Commodities": -0.15,
        },
    },
    "inflation_spike": {
        "name": "Inflation Spike",
        "description": "Persistent inflation, real asset outperformance",
        "shocks": {
            "US Equity": -0.10,
            "Intl Equity": -0.08,
            "US Bonds": -0.10,
            "REIT": -0.05,
            "Commodities": 0.15,
            "Technology": -0.15,
            "Consumer Defensive": -0.03,
            "Energy": 0.10,
        },
    },
    "tech_rotation": {
        "name": "Tech Sector Rotation",
        "description": "Rotation out of growth/tech into value/defensives",
        "shocks": {
            "Technology": -0.25,
            "Communication Services": -0.15,
            "Consumer Cyclical": -0.10,
            "Financial Services": 0.05,
            "Healthcare": 0.03,
            "Consumer Defensive": 0.05,
            "Utilities": 0.05,
            "US Bonds": 0.02,
        },
    },
}


def _get_holding_shock(holding: Holding, shocks: Dict[str, float], sector_override: str = "", asset_class_override: str = "") -> float:
    """Determine the shock % for a holding based on sector/asset class."""
    sector = sector_override or get_sector(holding.ticker)
    asset_class = asset_class_override or get_asset_class(holding.ticker)

    # Sector-level shock takes priority
    if sector in shocks:
        return shocks[sector]
    # Fall back to asset class
    if asset_class in shocks:
        return shocks[asset_class]
    # Default: assume equity-like
    return shocks.get("US Equity", -0.10)


def run_scenario(
    holdings: List[Holding],
    total_value: float,
    scenario_key: str,
    sector_map: Optional[Dict[str, str]] = None,
    asset_class_map: Optional[Dict[str, str]] = None,
) -> ScenarioResult:
    """Apply a stress scenario to portfolio holdings.

    Parameters
    ----------
    holdings : list of Holding
        Portfolio positions with current weights/values.
    total_value : float
        Total portfolio market value.
    scenario_key : str
        Key into ``SCENARIOS`` dict.
    sector_map : dict, optional
        Pre-loaded ticker → sector mapping (avoids yfinance calls).
    asset_class_map : dict, optional
        Pre-loaded ticker → asset class mapping.

    Returns
    -------
    ScenarioResult
    """
    scenario = SCENARIOS[scenario_key]
    shocks = scenario["shocks"]

    impacts = []
    for h in holdings:
        sec = (sector_map or {}).get(h.ticker, "")
        ac = (asset_class_map or {}).get(h.ticker, "")
        shock = _get_holding_shock(h, shocks, sec, ac)
        impact_usd = h.market_value * shock
        impacts.append({
            "ticker": h.ticker,
            "weight": h.weight,
            "shock_pct": shock,
            "impact_usd": impact_usd,
        })

    total_impact = sum(i["impact_usd"] for i in impacts)
    total_impact_pct = total_impact / total_value if total_value > 0 else 0

    sorted_impacts = sorted(impacts, key=lambda x: x["impact_usd"])
    worst = sorted_impacts[:3]
    best = sorted_impacts[-3:][::-1]

    return ScenarioResult(
        name=scenario["name"],
        description=scenario["description"],
        portfolio_impact_pct=total_impact_pct,
        portfolio_impact_usd=total_impact,
        worst_holdings=worst,
        best_holdings=best,
    )


def run_all_scenarios(
    holdings: List[Holding],
    total_value: float,
    sector_map: Optional[Dict[str, str]] = None,
    asset_class_map: Optional[Dict[str, str]] = None,
) -> List[ScenarioResult]:
    """Run all predefined stress scenarios."""
    return [run_scenario(holdings, total_value, key, sector_map, asset_class_map) for key in SCENARIOS]
