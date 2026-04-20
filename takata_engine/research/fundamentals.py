"""Fundamental data extraction and analysis from yfinance."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from takata_engine.data.feed_market import MarketDataFeed

logger = logging.getLogger(__name__)


def get_fundamentals(ticker: str) -> Dict[str, Any]:
    """Extract comprehensive fundamentals for a ticker.

    Returns a flat dict with all metrics needed for the equity research report.
    """
    t = yf.Ticker(ticker)
    info = t.info or {}

    def _g(key, default=None):
        val = info.get(key, default)
        return val if val is not None else default

    # Price data
    feed = MarketDataFeed(ticker, period="5y", interval="1d")
    try:
        daily = feed.load()
    except Exception:
        daily = pd.DataFrame()

    price = _g("currentPrice", _g("regularMarketPrice", 0))
    mkt_cap = _g("marketCap", 0)

    # ── Business overview ──
    fundamentals = {
        "ticker": ticker,
        "name": _g("shortName", ticker),
        "sector": _g("sector", "Unknown"),
        "industry": _g("industry", "Unknown"),
        "description": _g("longBusinessSummary", ""),
        "market_cap": mkt_cap,
        "market_cap_label": _format_cap(mkt_cap),
        "enterprise_value": _g("enterpriseValue", 0),
        "price": price,
        "currency": _g("currency", "USD"),
        "exchange": _g("exchange", ""),
        "employees": _g("fullTimeEmployees", 0),
    }

    # ── Valuation ──
    fundamentals.update({
        "pe_trailing": _g("trailingPE"),
        "pe_forward": _g("forwardPE"),
        "peg_ratio": _g("pegRatio"),
        "pb_ratio": _g("priceToBook"),
        "ps_ratio": _g("priceToSalesTrailing12Months"),
        "ev_ebitda": _g("enterpriseToEbitda"),
        "ev_revenue": _g("enterpriseToRevenue"),
    })

    # ── Profitability ──
    fundamentals.update({
        "revenue": _g("totalRevenue", 0),
        "revenue_growth": _g("revenueGrowth"),
        "gross_margin": _g("grossMargins"),
        "operating_margin": _g("operatingMargins"),
        "profit_margin": _g("profitMargins"),
        "ebitda": _g("ebitda", 0),
        "net_income": _g("netIncomeToCommon", 0),
        "eps_trailing": _g("trailingEps"),
        "eps_forward": _g("forwardEps"),
        "earnings_growth": _g("earningsGrowth"),
        "roe": _g("returnOnEquity"),
        "roa": _g("returnOnAssets"),
    })

    # ── Balance sheet ──
    fundamentals.update({
        "total_cash": _g("totalCash", 0),
        "total_debt": _g("totalDebt", 0),
        "debt_to_equity": _g("debtToEquity"),
        "current_ratio": _g("currentRatio"),
        "quick_ratio": _g("quickRatio"),
        "book_value": _g("bookValue"),
    })

    # ── Cash flow ──
    fundamentals.update({
        "operating_cf": _g("operatingCashflow", 0),
        "free_cf": _g("freeCashflow", 0),
        "fcf_yield": (_g("freeCashflow", 0) / mkt_cap * 100) if mkt_cap > 0 else 0,
    })

    # ── Dividends ──
    fundamentals.update({
        "dividend_rate": _g("dividendRate", 0),
        "dividend_yield": _g("dividendYield", 0),
        "payout_ratio": _g("payoutRatio"),
        "ex_dividend_date": _g("exDividendDate"),
    })

    # ── Analyst ──
    fundamentals.update({
        "target_high": _g("targetHighPrice"),
        "target_low": _g("targetLowPrice"),
        "target_mean": _g("targetMeanPrice"),
        "target_median": _g("targetMedianPrice"),
        "recommendation": _g("recommendationKey", ""),
        "recommendation_mean": _g("recommendationMean"),
        "num_analysts": _g("numberOfAnalystOpinions", 0),
    })

    # ── Price history stats ──
    if not daily.empty:
        close = daily["close"]
        fundamentals["price_52w_high"] = float(close.rolling(252).max().iloc[-1])
        fundamentals["price_52w_low"] = float(close.rolling(252).min().iloc[-1])
        fundamentals["price_vs_52w_high"] = (price / fundamentals["price_52w_high"] - 1) * 100
        fundamentals["return_1y"] = (close.iloc[-1] / close.iloc[-252] - 1) * 100 if len(close) >= 252 else None
        fundamentals["return_3y_ann"] = (
            ((close.iloc[-1] / close.iloc[-756]) ** (1/3) - 1) * 100
            if len(close) >= 756 else None
        )
        fundamentals["return_5y_ann"] = (
            ((close.iloc[-1] / close.iloc[0]) ** (1/5) - 1) * 100
            if len(close) >= 1200 else None
        )
        # Volatility
        fundamentals["volatility_1y"] = float(close.pct_change().tail(252).std() * np.sqrt(252) * 100)
    else:
        fundamentals.update({
            "price_52w_high": None, "price_52w_low": None, "price_vs_52w_high": None,
            "return_1y": None, "return_3y_ann": None, "return_5y_ann": None,
            "volatility_1y": None,
        })

    return fundamentals


def moat_score(f: Dict[str, Any]) -> Dict[str, Any]:
    """Competitive moat scoring (1-10) based on fundamentals.

    Evaluates: margins, returns on capital, market position, balance sheet.
    """
    scores = {}

    # Profit margin strength (0-2 points)
    pm = f.get("profit_margin") or 0
    scores["margins"] = min(2.0, pm * 10) if pm else 0

    # Return on equity (0-2 points)
    roe = f.get("roe") or 0
    scores["returns"] = min(2.0, roe * 8) if roe else 0

    # Revenue growth (0-2 points)
    rg = f.get("revenue_growth") or 0
    scores["growth"] = min(2.0, max(0, 1 + rg * 5))

    # Balance sheet (0-2 points)
    de = f.get("debt_to_equity")
    cr = f.get("current_ratio") or 0
    bs_score = 0
    if de is not None and de < 100:
        bs_score += 1
    if cr > 1.5:
        bs_score += 1
    scores["balance_sheet"] = bs_score

    # Market cap / scale (0-2 points)
    mc = f.get("market_cap", 0)
    if mc > 200e9:
        scores["scale"] = 2.0
    elif mc > 50e9:
        scores["scale"] = 1.5
    elif mc > 10e9:
        scores["scale"] = 1.0
    else:
        scores["scale"] = 0.5

    total = sum(scores.values())
    return {"total": round(total, 1), "breakdown": scores}


def _format_cap(cap: float) -> str:
    if cap >= 1e12:
        return f"${cap/1e12:.1f}T"
    if cap >= 1e9:
        return f"${cap/1e9:.1f}B"
    if cap >= 1e6:
        return f"${cap/1e6:.0f}M"
    return f"${cap:,.0f}"


def _format_big(val: float) -> str:
    if val is None or val == 0:
        return "N/A"
    if abs(val) >= 1e9:
        return f"${val/1e9:.1f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.0f}M"
    return f"${val:,.0f}"


def peer_comparison(ticker: str, peers: Optional[List[str]] = None) -> pd.DataFrame:
    """Compare valuation metrics across peers.

    If no peers provided, uses sector peers from yfinance.
    """
    if peers is None:
        # Default peer sets by sector
        info = (yf.Ticker(ticker).info or {})
        sector = info.get("sector", "")
        peers = _default_peers(ticker, sector)

    tickers = [ticker] + [p for p in peers if p != ticker]

    rows = []
    for t in tickers[:8]:  # cap at 8 to keep fast
        try:
            f = get_fundamentals(t)
            rows.append({
                "ticker": t,
                "name": f["name"][:20],
                "mkt_cap": f["market_cap_label"],
                "PE": f.get("pe_trailing"),
                "Fwd PE": f.get("pe_forward"),
                "PB": f.get("pb_ratio"),
                "EV/EBITDA": f.get("ev_ebitda"),
                "Margin": f.get("profit_margin"),
                "ROE": f.get("roe"),
                "Rev Growth": f.get("revenue_growth"),
            })
        except Exception as e:
            logger.warning("Peer comparison failed for %s: %s", t, e)

    return pd.DataFrame(rows)


def _default_peers(ticker: str, sector: str) -> List[str]:
    """Return default peer tickers by sector."""
    peers = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "CRM"],
        "Financial Services": ["JPM", "BAC", "GS", "MS", "BRK-B", "V"],
        "Healthcare": ["UNH", "JNJ", "PFE", "ABBV", "LLY", "MRK"],
        "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX"],
        "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "T", "VZ"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC"],
        "Consumer Defensive": ["PG", "KO", "PEP", "COST", "WMT", "CL"],
        "Real Estate": ["O", "AMT", "PLD", "SPG", "EQIX", "PSA"],
    }
    peer_list = peers.get(sector, ["SPY", "QQQ"])
    return [p for p in peer_list if p != ticker][:5]
