"""Research Analyst agent — Claude-enhanced equity research.

Takes the signal engine's quantitative research output and generates
a Goldman Sachs-quality investment thesis with narrative depth.

Uses Sonnet for balanced depth/speed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from takata_core.agents.llm_client import call_claude, MODEL_BALANCED

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior equity research analyst at a top-tier investment bank.

Given quantitative data about a stock (fundamentals, technicals, sector positioning),
generate a professional investment thesis.

Your output should be:
- Institutional quality — like a Goldman Sachs or JPMorgan research note
- Data-driven with specific numbers cited
- Clear on conviction level and catalysts
- Honest about risks — no promotional fluff

The trader manages two accounts:
- Takata Holdings (personal): active trading + long-term positions
- Sympatheia Advisory (business): client portfolios, conservative

Respond with JSON:
{
  "rating": "Strong Buy" | "Buy" | "Hold" | "Sell" | "Strong Sell",
  "target_price": 0.0,
  "conviction": "high" | "medium" | "low",
  "thesis_headline": "One-line investment thesis",
  "bull_case": "2-3 sentences on why this stock goes up",
  "bear_case": "2-3 sentences on why this stock goes down",
  "catalysts": ["catalyst1", "catalyst2"],
  "risks": ["risk1", "risk2"],
  "technical_view": "1-2 sentences on chart setup",
  "sector_context": "How does this fit in the sector rotation?",
  "position_sizing": "Suggested allocation (% of portfolio)",
  "time_horizon": "Days / Weeks / Months / Quarters",
  "covered_call_overlay": "If holding, what strike/DTE for income overlay?",
  "full_analysis": "3-4 paragraph detailed analysis"
}"""


def deep_research(
    ticker: str,
    fundamentals: Dict[str, Any] = None,
    technicals: Dict[str, Any] = None,
    sector_data: Dict[str, Any] = None,
    macro_context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Generate deep equity research report enhanced by Claude.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    fundamentals : dict, optional
        Fundamental data (P/E, revenue, margins, moat score, etc).
    technicals : dict, optional
        Technical analysis (trend, RSI, support/resistance, patterns).
    sector_data : dict, optional
        Sector rotation context (momentum, relative strength).
    macro_context : dict, optional
        Current macro regime and instrument biases.

    Returns
    -------
    dict
        Full research analysis with rating, thesis, bull/bear cases.
    """
    prompt = f"""Generate an institutional-quality research report for **{ticker}**.

## Fundamental Data
{json.dumps(fundamentals or {}, indent=2, default=str)}

## Technical Analysis
{json.dumps(technicals or {}, indent=2, default=str)}

## Sector Context
{json.dumps(sector_data or {}, indent=2, default=str)}

## Macro Environment
{json.dumps(macro_context or {}, indent=2, default=str)}

Respond with ONLY the JSON."""

    result = call_claude(
        system=SYSTEM_PROMPT,
        user_prompt=prompt,
        model=MODEL_BALANCED,
        max_tokens=2000,
    )

    if isinstance(result, dict) and not result.get("_parse_error"):
        result["ticker"] = ticker
        return result

    return {
        "ticker": ticker,
        "rating": "Hold",
        "conviction": "low",
        "thesis_headline": f"Deep analysis for {ticker} unavailable",
        "full_analysis": result.get("raw_text", str(result))[:500] if isinstance(result, dict) else str(result)[:500],
    }
