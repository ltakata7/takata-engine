"""Multi-factor scoring — value, quality, momentum, growth, sentiment."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from takata_core.data.feed_market import MarketDataFeed

logger = logging.getLogger(__name__)


def _get_info(ticker: str) -> dict:
    """Fetch yfinance ticker info with error handling."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as e:
        logger.warning("Could not fetch info for %s: %s", ticker, e)
        return {}


def _safe_get(info: dict, key: str, default: float = np.nan) -> float:
    val = info.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ── Individual factor scores (0-100, higher is better) ──────────────


def value_score(info: dict) -> Dict[str, Any]:
    """Value factor: PE, PB, dividend yield, EV/EBITDA."""
    pe = _safe_get(info, "trailingPE")
    pb = _safe_get(info, "priceToBook")
    div_yield = _safe_get(info, "dividendYield", 0) * 100  # to percent
    ev_ebitda = _safe_get(info, "enterpriseToEbitda")

    scores = []
    details = {}

    # Lower PE is better (invert and cap)
    if not np.isnan(pe) and pe > 0:
        s = max(0, min(100, 100 - (pe - 10) * 3))
        scores.append(s)
        details["PE"] = round(pe, 1)
    # Lower PB is better
    if not np.isnan(pb) and pb > 0:
        s = max(0, min(100, 100 - (pb - 1) * 20))
        scores.append(s)
        details["PB"] = round(pb, 2)
    # Higher dividend yield is better
    if div_yield > 0:
        s = min(100, div_yield * 25)
        scores.append(s)
        details["Div Yield"] = f"{div_yield:.2f}%"
    # Lower EV/EBITDA is better
    if not np.isnan(ev_ebitda) and ev_ebitda > 0:
        s = max(0, min(100, 100 - (ev_ebitda - 8) * 5))
        scores.append(s)
        details["EV/EBITDA"] = round(ev_ebitda, 1)

    composite = float(np.mean(scores)) if scores else 50.0
    return {"score": round(composite, 1), "details": details}


def quality_score(info: dict) -> Dict[str, Any]:
    """Quality factor: ROE, profit margins, debt/equity, current ratio."""
    roe = _safe_get(info, "returnOnEquity", 0) * 100
    margin = _safe_get(info, "profitMargins", 0) * 100
    de = _safe_get(info, "debtToEquity")
    current = _safe_get(info, "currentRatio")

    scores = []
    details = {}

    # Higher ROE is better
    if roe > 0:
        s = min(100, roe * 4)
        scores.append(s)
        details["ROE"] = f"{roe:.1f}%"
    # Higher margin is better
    if margin > 0:
        s = min(100, margin * 3)
        scores.append(s)
        details["Profit Margin"] = f"{margin:.1f}%"
    # Lower D/E is better
    if not np.isnan(de):
        s = max(0, min(100, 100 - de * 0.5))
        scores.append(s)
        details["D/E"] = round(de, 1)
    # Higher current ratio is better (but diminishing)
    if not np.isnan(current) and current > 0:
        s = min(100, current * 40)
        scores.append(s)
        details["Current Ratio"] = round(current, 2)

    composite = float(np.mean(scores)) if scores else 50.0
    return {"score": round(composite, 1), "details": details}


def momentum_score(ticker: str) -> Dict[str, Any]:
    """Momentum factor: 1M, 3M, 6M, 12M returns and relative strength."""
    try:
        feed = MarketDataFeed(ticker, period="1y", interval="1d")
        df = feed.load()
        close = df["close"]
    except Exception:
        return {"score": 50.0, "details": {}}

    details = {}
    scores = []
    n = len(close)

    periods = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
    for label, bars in periods.items():
        if n > bars:
            ret = (close.iloc[-1] / close.iloc[-bars] - 1) * 100
            # Positive momentum scores higher
            s = max(0, min(100, 50 + ret * 2))
            scores.append(s)
            details[label] = f"{ret:+.1f}%"

    composite = float(np.mean(scores)) if scores else 50.0
    return {"score": round(composite, 1), "details": details}


def growth_score(info: dict) -> Dict[str, Any]:
    """Growth factor: revenue growth, earnings growth, forward estimates."""
    rev_growth = _safe_get(info, "revenueGrowth", 0) * 100
    earn_growth = _safe_get(info, "earningsGrowth", 0) * 100
    fwd_pe = _safe_get(info, "forwardPE")
    trail_pe = _safe_get(info, "trailingPE")

    scores = []
    details = {}

    # Higher revenue growth is better
    if rev_growth != 0:
        s = max(0, min(100, 50 + rev_growth * 2))
        scores.append(s)
        details["Rev Growth"] = f"{rev_growth:+.1f}%"
    # Higher earnings growth is better
    if earn_growth != 0:
        s = max(0, min(100, 50 + earn_growth * 1.5))
        scores.append(s)
        details["EPS Growth"] = f"{earn_growth:+.1f}%"
    # PEG-like: forward PE < trailing PE suggests growth
    if not np.isnan(fwd_pe) and not np.isnan(trail_pe) and trail_pe > 0:
        peg_proxy = fwd_pe / trail_pe
        s = max(0, min(100, 100 - peg_proxy * 50))
        scores.append(s)
        details["Fwd/Trail PE"] = f"{peg_proxy:.2f}"

    composite = float(np.mean(scores)) if scores else 50.0
    return {"score": round(composite, 1), "details": details}


def sentiment_score(info: dict) -> Dict[str, Any]:
    """Sentiment factor: analyst ratings, short interest, institutional ownership."""
    rec = _safe_get(info, "recommendationMean")  # 1=strong buy, 5=sell
    target = _safe_get(info, "targetMeanPrice")
    current = _safe_get(info, "currentPrice")
    short_pct = _safe_get(info, "shortPercentOfFloat", 0) * 100
    inst_pct = _safe_get(info, "heldPercentInstitutions", 0) * 100

    scores = []
    details = {}

    # Analyst recommendation (1=best, 5=worst)
    if not np.isnan(rec) and rec > 0:
        s = max(0, min(100, (5 - rec) * 25))
        scores.append(s)
        labels = {1: "Strong Buy", 2: "Buy", 3: "Hold", 4: "Sell", 5: "Strong Sell"}
        details["Analyst"] = labels.get(round(rec), f"{rec:.1f}")

    # Price target upside
    if not np.isnan(target) and not np.isnan(current) and current > 0:
        upside = (target / current - 1) * 100
        s = max(0, min(100, 50 + upside * 2))
        scores.append(s)
        details["Target Upside"] = f"{upside:+.1f}%"

    # Low short interest is better
    if short_pct > 0:
        s = max(0, min(100, 100 - short_pct * 5))
        scores.append(s)
        details["Short %"] = f"{short_pct:.1f}%"

    # High institutional ownership (moderate is best)
    if inst_pct > 0:
        s = min(100, inst_pct * 1.2)
        scores.append(s)
        details["Inst Own"] = f"{inst_pct:.0f}%"

    composite = float(np.mean(scores)) if scores else 50.0
    return {"score": round(composite, 1), "details": details}


def compute_all_factors(ticker: str) -> Dict[str, Any]:
    """Compute all factor scores for a single ticker.

    Returns
    -------
    dict
        Keys: ``ticker``, ``value``, ``quality``, ``momentum``, ``growth``,
        ``sentiment``, ``composite`` (weighted average 1-100).
    """
    info = _get_info(ticker)

    factors = {
        "value": value_score(info),
        "quality": quality_score(info),
        "momentum": momentum_score(ticker),
        "growth": growth_score(info),
        "sentiment": sentiment_score(info),
    }

    # Composite: equal weight across factors
    factor_scores = [f["score"] for f in factors.values()]
    composite = float(np.mean(factor_scores))

    return {
        "ticker": ticker,
        **factors,
        "composite": round(composite, 1),
        "sector": info.get("sector", "Unknown"),
        "name": info.get("shortName", ticker),
    }
