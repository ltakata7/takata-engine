"""Portfolio construction — multiple strategy styles with risk-profile adaptation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AllocationSlice:
    """A single allocation within the portfolio."""
    category: str
    role: str       # "core" or "satellite"
    etf: str
    etf_name: str
    target_pct: float
    rationale: str


@dataclass
class PortfolioPlan:
    """Complete portfolio construction plan."""
    profile_age: int
    profile_risk: str
    profile_amount: float
    style: str
    style_description: str
    allocations: List[AllocationSlice]
    expected_return_low: float
    expected_return_high: float
    rebalancing_rules: List[str]
    tax_notes: List[str]
    dca_plan: str


# ── Available styles ──

STYLES = {
    "vanguard": "Vanguard Passive — Low-cost, globally diversified, age-based",
    "all_weather": "All-Weather (Bridgewater) — Perform in any economic regime",
    "risk_parity": "Risk Parity — Equal risk contribution across asset classes",
    "income": "Income / Dividend — Maximize yield and cash flow",
    "factor": "Factor-Based (AQR) — Value + Momentum + Quality smart-beta tilts",
}

RISK_PROFILES = ["aggressive", "growth", "moderate", "conservative", "very_conservative"]


def _determine_risk_profile(age: int, risk: str) -> str:
    risk = risk.lower()
    if risk in RISK_PROFILES:
        return risk
    if age < 30: base = "aggressive"
    elif age < 40: base = "growth"
    elif age < 50: base = "moderate"
    elif age < 60: base = "conservative"
    else: base = "very_conservative"
    if risk in ("high", "aggressive"):
        idx = max(0, RISK_PROFILES.index(base) - 1)
        return RISK_PROFILES[idx]
    elif risk in ("low", "conservative"):
        idx = min(len(RISK_PROFILES) - 1, RISK_PROFILES.index(base) + 1)
        return RISK_PROFILES[idx]
    return base


# ── Risk multipliers per profile ──
# Scales how much equity/risk each profile takes within a strategy
RISK_SCALE = {
    "aggressive": 1.0,
    "growth": 0.85,
    "moderate": 0.70,
    "conservative": 0.50,
    "very_conservative": 0.35,
}


def _normalize(allocs: List[AllocationSlice]) -> List[AllocationSlice]:
    total = sum(a.target_pct for a in allocs)
    if total > 0:
        for a in allocs:
            a.target_pct = round(a.target_pct / total * 100, 1)
    return allocs


def _build_vanguard(risk: str, amount: float) -> tuple:
    """Vanguard passive: globally diversified, low cost."""
    s = RISK_SCALE[risk]
    eq = s * 90
    bond = (1 - s) * 70 + 5
    alt = 100 - eq - bond

    allocs = [
        AllocationSlice("US Equity", "core", "VTI", "Vanguard Total Stock Market",
                        eq * 0.60, "Broad US market, 0.03% ER"),
        AllocationSlice("Intl Developed", "core", "VXUS", "Vanguard Total International",
                        eq * 0.30, "Developed markets diversification"),
        AllocationSlice("Emerging Markets", "core", "VWO", "Vanguard Emerging Markets",
                        eq * 0.10, "EM growth exposure"),
        AllocationSlice("US Bonds", "core", "BND", "Vanguard Total Bond Market",
                        bond * 0.60, "Core fixed income"),
        AllocationSlice("TIPS", "core", "VTIP", "Vanguard Short-Term TIPS",
                        bond * 0.20, "Inflation protection"),
        AllocationSlice("Intl Bonds", "core", "BNDX", "Vanguard Total Intl Bond",
                        bond * 0.20, "Global fixed income"),
        AllocationSlice("REITs", "satellite", "VNQ", "Vanguard Real Estate",
                        alt * 0.50, "Real asset income"),
        AllocationSlice("Gold", "satellite", "IAU", "iShares Gold Trust",
                        alt * 0.50, "Tail risk hedge"),
    ]
    allocs = [a for a in allocs if a.target_pct >= 2]
    ret_mid = eq / 100 * 0.08 + bond / 100 * 0.04 + alt / 100 * 0.06
    return _normalize(allocs), ret_mid


def _build_all_weather(risk: str, amount: float) -> tuple:
    """Bridgewater All-Weather: designed for any economic environment.
    Base: 30% stocks, 40% long bonds, 15% medium bonds, 7.5% gold, 7.5% commodities.
    Adjusted by risk profile."""
    s = RISK_SCALE[risk]

    # All-weather base with risk scaling
    stocks = 30 * (1 + (s - 0.7) * 0.8)   # 22-36%
    long_bonds = 40 * (1 - (s - 0.7) * 0.3)  # 36-44%
    med_bonds = 15 * (1 - (s - 0.7) * 0.2)
    gold = 7.5 * (1 + (s - 0.7) * 0.3)
    commodities = 100 - stocks - long_bonds - med_bonds - gold

    allocs = [
        AllocationSlice("US Equity", "core", "VTI", "Vanguard Total Stock Market",
                        stocks * 0.60, "Equity growth engine — performs in growth/inflation"),
        AllocationSlice("Intl Equity", "core", "VXUS", "Vanguard Total International",
                        stocks * 0.40, "Global equity diversification"),
        AllocationSlice("Long-Term Bonds", "core", "TLT", "iShares 20+ Year Treasury",
                        long_bonds, "Deflation/recession protection — anti-correlated to equity"),
        AllocationSlice("Medium Bonds", "core", "IEF", "iShares 7-10 Year Treasury",
                        med_bonds, "Moderate duration — balanced rate sensitivity"),
        AllocationSlice("Gold", "core", "GLD", "SPDR Gold Shares",
                        gold, "Inflation hedge — uncorrelated to stocks and bonds"),
        AllocationSlice("Commodities", "core", "DJP", "iPath Bloomberg Commodity",
                        max(commodities, 2), "Inflation/growth surprise protection"),
    ]
    allocs = [a for a in allocs if a.target_pct >= 2]
    ret_mid = stocks / 100 * 0.07 + (long_bonds + med_bonds) / 100 * 0.04 + (gold + max(commodities, 0)) / 100 * 0.05
    return _normalize(allocs), ret_mid


def _build_risk_parity(risk: str, amount: float) -> tuple:
    """Risk Parity: equal risk contribution from each asset class.
    Bonds get higher allocation (lower vol) to match equity risk budget."""
    s = RISK_SCALE[risk]

    # Risk parity weights (inverse volatility)
    # Equity vol ~16%, Bond vol ~6%, Gold vol ~15%, REIT vol ~18%
    inv_vol = {"equity": 1/16, "bonds": 1/6, "gold": 1/15, "reit": 1/18}
    total_inv = sum(inv_vol.values())
    base_weights = {k: v / total_inv * 100 for k, v in inv_vol.items()}

    # Scale equity up/down by risk profile
    eq_adj = base_weights["equity"] * (0.7 + s * 0.6)
    bond_adj = base_weights["bonds"] * (1.3 - s * 0.3)
    gold_adj = base_weights["gold"]
    reit_adj = base_weights["reit"]

    allocs = [
        AllocationSlice("US Equity", "core", "USMV", "iShares Min Volatility",
                        eq_adj * 0.50, "Low-vol equity — matches risk parity framework"),
        AllocationSlice("Intl Equity", "core", "EFAV", "iShares Intl Min Vol",
                        eq_adj * 0.30, "International low-vol equity"),
        AllocationSlice("EM Equity", "core", "EEMV", "iShares EM Min Vol",
                        eq_adj * 0.20, "Emerging markets low-vol"),
        AllocationSlice("US Aggregate Bonds", "core", "AGG", "iShares Core US Aggregate",
                        bond_adj * 0.40, "Broad fixed income — high risk parity weight"),
        AllocationSlice("Long Treasury", "core", "TLT", "iShares 20+ Year Treasury",
                        bond_adj * 0.30, "Duration exposure for deflation protection"),
        AllocationSlice("TIPS", "core", "TIP", "iShares TIPS Bond",
                        bond_adj * 0.30, "Real return protection"),
        AllocationSlice("Gold", "core", "GLD", "SPDR Gold Shares",
                        gold_adj, "Uncorrelated risk diversifier"),
        AllocationSlice("REITs", "satellite", "VNQ", "Vanguard Real Estate",
                        reit_adj, "Real asset risk contribution"),
    ]
    allocs = [a for a in allocs if a.target_pct >= 2]
    ret_mid = eq_adj / 100 * 0.07 + bond_adj / 100 * 0.035 + gold_adj / 100 * 0.04 + reit_adj / 100 * 0.06
    return _normalize(allocs), ret_mid


def _build_income(risk: str, amount: float) -> tuple:
    """Income/Dividend: maximize yield and cash flow."""
    s = RISK_SCALE[risk]

    div_eq = 35 * (0.8 + s * 0.4)    # 28-48%
    bonds = 40 * (1.2 - s * 0.4)     # 24-48%
    reit = 15 * (0.8 + s * 0.3)
    pref = 100 - div_eq - bonds - reit

    allocs = [
        AllocationSlice("Dividend Equity", "core", "SCHD", "Schwab US Dividend Equity",
                        div_eq * 0.40, "Quality dividend growth — 3.5% yield, low turnover"),
        AllocationSlice("High Dividend", "core", "VYM", "Vanguard High Dividend Yield",
                        div_eq * 0.30, "Broad high-yield equity — 3% yield"),
        AllocationSlice("Intl Dividend", "core", "VYMI", "Vanguard Intl High Dividend",
                        div_eq * 0.30, "International income — 4%+ yield"),
        AllocationSlice("Corporate Bonds", "core", "LQD", "iShares Investment Grade",
                        bonds * 0.35, "Investment grade corporates — 5% yield"),
        AllocationSlice("High Yield Bonds", "satellite", "HYG", "iShares High Yield",
                        bonds * 0.25, "High yield credit — 7%+ yield, higher risk"),
        AllocationSlice("Short-Term Bonds", "core", "SHY", "iShares 1-3 Year Treasury",
                        bonds * 0.40, "Capital preservation, low duration risk"),
        AllocationSlice("REITs", "core", "O", "Realty Income Corporation",
                        reit * 0.50, "Monthly dividend REIT — 5.5% yield"),
        AllocationSlice("REIT Index", "core", "VNQ", "Vanguard Real Estate",
                        reit * 0.50, "Diversified REIT exposure — 4% yield"),
        AllocationSlice("Preferred Stock", "satellite", "PFF", "iShares Preferred & Income",
                        max(pref, 2), "Preferred shares — 6%+ yield, bond-like"),
    ]
    allocs = [a for a in allocs if a.target_pct >= 2]
    # Income portfolio expected yield ~4-5%, total return lower
    ret_mid = div_eq / 100 * 0.07 + bonds / 100 * 0.045 + reit / 100 * 0.06 + max(pref, 0) / 100 * 0.055
    return _normalize(allocs), ret_mid


def _build_factor(risk: str, amount: float) -> tuple:
    """Factor-Based (AQR-style): Value + Momentum + Quality tilts."""
    s = RISK_SCALE[risk]

    equity = 70 * (0.7 + s * 0.5)   # 49-84%
    bonds = 100 - equity - 10        # remainder minus alts
    alts = 10

    allocs = [
        AllocationSlice("US Value", "core", "VTV", "Vanguard Value",
                        equity * 0.20, "Value factor — low P/E, high book value stocks"),
        AllocationSlice("US Momentum", "core", "MTUM", "iShares MSCI USA Momentum",
                        equity * 0.20, "Momentum factor — winners keep winning"),
        AllocationSlice("US Quality", "core", "QUAL", "iShares MSCI USA Quality",
                        equity * 0.20, "Quality factor — high ROE, stable earnings"),
        AllocationSlice("US Min Vol", "satellite", "USMV", "iShares Min Volatility",
                        equity * 0.15, "Low-vol factor — defensive equity"),
        AllocationSlice("US Small Value", "satellite", "VBR", "Vanguard Small Cap Value",
                        equity * 0.10, "Small-cap value — highest historical premium"),
        AllocationSlice("Intl Factor", "core", "IQLT", "iShares Intl Quality",
                        equity * 0.15, "International quality tilt"),
        AllocationSlice("US Aggregate", "core", "AGG", "iShares Core Aggregate",
                        bonds * 0.60, "Core bond ballast"),
        AllocationSlice("TIPS", "core", "TIP", "iShares TIPS Bond",
                        bonds * 0.40, "Real return protection"),
        AllocationSlice("Gold", "satellite", "GLD", "SPDR Gold Shares",
                        alts * 0.50, "Macro hedge"),
        AllocationSlice("Managed Futures", "satellite", "DBMF", "iMGP DBi Managed Futures",
                        alts * 0.50, "Trend-following — uncorrelated alpha source"),
    ]
    allocs = [a for a in allocs if a.target_pct >= 2]
    ret_mid = equity / 100 * 0.09 + bonds / 100 * 0.04 + alts / 100 * 0.06  # factor premium adds ~1%
    return _normalize(allocs), ret_mid


# ── Strategy builders map ──
STRATEGY_BUILDERS = {
    "vanguard": _build_vanguard,
    "all_weather": _build_all_weather,
    "risk_parity": _build_risk_parity,
    "income": _build_income,
    "factor": _build_factor,
}


class PortfolioConstructor:
    """Builds a portfolio using a chosen strategy style and risk profile.

    Parameters
    ----------
    age : int
        Investor age.
    amount : float
        Investment amount in USD.
    risk : str
        Risk tolerance.
    style : str
        Strategy style: ``"vanguard"``, ``"all_weather"``, ``"risk_parity"``,
        ``"income"``, ``"factor"``.
    """

    def __init__(self, age: int = 35, amount: float = 100000,
                 risk: str = "moderate", style: str = "vanguard"):
        self.age = age
        self.amount = amount
        self.risk = risk
        self.style = style.lower()
        self._plan: Optional[PortfolioPlan] = None

    def build(self) -> PortfolioPlan:
        profile_key = _determine_risk_profile(self.age, self.risk)
        builder = STRATEGY_BUILDERS.get(self.style, _build_vanguard)
        allocs, ret_mid = builder(profile_key, self.amount)

        expected_low = ret_mid - 0.03
        expected_high = ret_mid + 0.03

        rebalancing = {
            "vanguard": [
                "Rebalance when any allocation drifts >5% from target",
                "Calendar rebalance: review quarterly, execute if needed",
                "Tax-loss harvest during rebalancing when possible",
                "Use new contributions to rebalance before selling",
            ],
            "all_weather": [
                "Rebalance quarterly to maintain target risk parity",
                "Rebalance after any 10%+ drawdown in a single asset class",
                "Do NOT rebalance during volatility spikes — wait for VIX < 25",
                "Annual strategic review of macro regime assumptions",
            ],
            "risk_parity": [
                "Monthly risk contribution check — rebalance if any asset > 30% of total risk",
                "Quarterly full rebalance to inverse-volatility weights",
                "Adjust weights if 60-day realized vol changes >50% from assumption",
                "Annual review of correlation assumptions",
            ],
            "income": [
                "Reinvest dividends to maintain target weights (DRIP)",
                "Rebalance semi-annually — income portfolios need less frequent adjustment",
                "Review credit quality quarterly — reduce HY if spreads widen >200bp",
                "Tax-manage: harvest losses on bond positions near year-end",
            ],
            "factor": [
                "Quarterly factor momentum review — rotate underperforming factors",
                "Rebalance when any factor tilt drifts >3% from target",
                "Annual factor premium review — confirm value/momentum/quality still positive",
                "Monitor factor crowding via AUM flows into factor ETFs",
            ],
        }

        tax = [
            "Hold bonds and REITs in tax-advantaged accounts (IRA/401k)",
            "Hold equity ETFs in taxable accounts (qualified dividends, lower turnover)",
            "Harvest losses on individual positions to offset gains",
            "Consider tax-lot selection (specific identification) for optimal harvesting",
        ]

        monthly = self.amount / 12
        dca = (
            f"Invest ${monthly:,.0f}/month over 12 months. "
            f"Allocate each contribution proportionally to target weights. "
            f"If market drops >10%, accelerate deployment (invest 2x monthly amount)."
        )

        self._plan = PortfolioPlan(
            profile_age=self.age,
            profile_risk=profile_key,
            profile_amount=self.amount,
            style=self.style,
            style_description=STYLES.get(self.style, ""),
            allocations=allocs,
            expected_return_low=round(expected_low * 100, 1),
            expected_return_high=round(expected_high * 100, 1),
            rebalancing_rules=rebalancing.get(self.style, rebalancing["vanguard"]),
            tax_notes=tax,
            dca_plan=dca,
        )
        return self._plan

    def format_report(self) -> str:
        if self._plan is None:
            self.build()
        p = self._plan
        w = 70
        lines = [
            "=" * w,
            f"  PORTFOLIO CONSTRUCTION — {p.style.upper().replace('_', ' ')}",
            f"  {p.style_description}",
            f"  Profile: Age {p.profile_age} | Risk: {p.profile_risk.upper()} | "
            f"Amount: ${p.profile_amount:,.0f}",
            "=" * w, "",
            "ASSET ALLOCATION", "-" * w,
            f"  {'Category':20s} {'Role':10s} {'ETF':6s} {'Target':>7s} {'Amount':>12s}",
        ]
        for a in p.allocations:
            amt = p.profile_amount * a.target_pct / 100
            lines.append(f"  {a.category:20s} {a.role:10s} {a.etf:6s} {a.target_pct:>6.1f}% ${amt:>11,.0f}")

        lines.extend([
            "", f"EXPECTED RETURN RANGE: {p.expected_return_low:.1f}% — {p.expected_return_high:.1f}% annualized",
            "", "REBALANCING RULES", "-" * w,
        ])
        for rule in p.rebalancing_rules:
            lines.append(f"  * {rule}")
        lines.extend(["", "DCA PLAN", "-" * w, f"  {p.dca_plan}", "", "=" * w])
        return "\n".join(lines)
