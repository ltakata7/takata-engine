"""Portfolio concentration analysis — sector, asset class, HHI."""

from __future__ import annotations

import logging
from typing import Dict, List

import yfinance as yf

from takata_core.risk_portfolio.portfolio import Holding

logger = logging.getLogger(__name__)

# Sector classification cache
_SECTOR_CACHE: Dict[str, str] = {}
_ASSET_CLASS_CACHE: Dict[str, str] = {}

# Known ETF asset class mappings
_ETF_ASSET_CLASS = {
    "SPY": "US Equity", "QQQ": "US Equity", "IWM": "US Equity",
    "DIA": "US Equity", "VOO": "US Equity", "VTI": "US Equity",
    "SCHD": "US Equity", "VIG": "US Equity", "VYM": "US Equity",
    "EFA": "Intl Equity", "EEM": "Intl Equity", "VXUS": "Intl Equity",
    "TLT": "US Bonds", "IEF": "US Bonds", "AGG": "US Bonds",
    "BND": "US Bonds", "SHY": "US Bonds", "LQD": "US Bonds",
    "GLD": "Commodities", "SLV": "Commodities", "IAU": "Commodities",
    "O": "REIT", "VNQ": "REIT", "SCHH": "REIT",
}


def _get_ticker_info(ticker: str) -> dict:
    """Fetch ticker info from yfinance (cached)."""
    try:
        t = yf.Ticker(ticker)
        return t.info or {}
    except Exception:
        return {}


def get_sector(ticker: str) -> str:
    """Get sector for a ticker."""
    if ticker in _SECTOR_CACHE:
        return _SECTOR_CACHE[ticker]

    if ticker in _ETF_ASSET_CLASS:
        _SECTOR_CACHE[ticker] = _ETF_ASSET_CLASS[ticker]
        return _SECTOR_CACHE[ticker]

    info = _get_ticker_info(ticker)
    sector = info.get("sector", "Unknown")
    _SECTOR_CACHE[ticker] = sector
    return sector


def get_asset_class(ticker: str) -> str:
    """Get asset class for a ticker."""
    if ticker in _ETF_ASSET_CLASS:
        return _ETF_ASSET_CLASS[ticker]

    info = _get_ticker_info(ticker)
    quote_type = info.get("quoteType", "")

    if quote_type == "ETF":
        category = info.get("category", "")
        if "bond" in category.lower() or "fixed" in category.lower():
            return "US Bonds"
        if "international" in category.lower() or "foreign" in category.lower():
            return "Intl Equity"
        return "US Equity"
    elif info.get("sector") == "Real Estate":
        return "REIT"
    else:
        return "US Equity"


def sector_breakdown(holdings: List[Holding]) -> Dict[str, float]:
    """Compute sector allocation weights."""
    sectors: Dict[str, float] = {}
    for h in holdings:
        sector = get_sector(h.ticker)
        sectors[sector] = sectors.get(sector, 0) + h.weight
    return dict(sorted(sectors.items(), key=lambda x: -x[1]))


def asset_class_breakdown(holdings: List[Holding]) -> Dict[str, float]:
    """Compute asset class allocation weights."""
    classes: Dict[str, float] = {}
    for h in holdings:
        ac = get_asset_class(h.ticker)
        classes[ac] = classes.get(ac, 0) + h.weight
    return dict(sorted(classes.items(), key=lambda x: -x[1]))


def top_holdings_concentration(holdings: List[Holding], n: int = 5) -> float:
    """Weight of top N holdings."""
    sorted_h = sorted(holdings, key=lambda h: -h.weight)
    return sum(h.weight for h in sorted_h[:n])


def herfindahl_index(holdings: List[Holding]) -> float:
    """Herfindahl-Hirschman Index (0-1). Higher = more concentrated."""
    return sum(h.weight ** 2 for h in holdings)
