"""Macro Flash agent — intraday macro situation report.

Generates a quick macro update any time during the trading day.
Combines latest FRED, BCB, DI curve, and market data into
a 2-3 paragraph actionable summary.

Uses Haiku for speed — traders want this NOW.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict

from takata_engine.agents.llm_client import call_claude, MODEL_FAST

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a macro strategist providing flash updates to a Brazilian futures day trader.

Generate a CONCISE (2-3 paragraph) macro situation report for right now.

The trader day-trades:
- WDO (Mini Dollar B3, R$10/point) — driven by USD/BRL, carry trade, risk appetite
- MES (Micro E-mini S&P, $5/point) — driven by US equity sentiment, tech, Fed

Focus on:
1. What changed since market open? Any surprises?
2. Current risk appetite and flow direction
3. Specific impact on WDO and MES positioning

PTAX windows (10-11, 13-14 BRT) matter for WDO — flag if currently in one.
DI curve moves matter for carry trade — flag significant changes.

Respond with JSON:
{
  "headline": "One-line flash update",
  "wdo_impact": "bullish" | "bearish" | "neutral",
  "mes_impact": "bullish" | "bearish" | "neutral",
  "urgency": "low" | "normal" | "high" | "critical",
  "flash_report": "2-3 paragraph macro update",
  "action_items": ["item1", "item2"],
  "key_levels_to_watch": {"wdo": [level1], "mes": [level1]}
}

Be concise and actionable. No filler."""


def generate_flash(
    macro_data: Dict[str, Any] = None,
    market_data: Dict[str, Any] = None,
    briefing_context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Generate an intraday macro flash report.

    Parameters
    ----------
    macro_data : dict, optional
        Latest FRED + BCB indicators.
    market_data : dict, optional
        Latest WDO/MES prices, DI curve, bridge data.
    briefing_context : dict, optional
        Today's pre-market briefing for comparison.

    Returns
    -------
    dict
        Flash report with impact assessment and action items.
    """
    # Gather fresh data if not provided
    if macro_data is None:
        try:
            from takata_engine.agents.premarket_briefing import _gather_macro_data
            macro_data = _gather_macro_data()
        except Exception as e:
            logger.warning("Could not gather macro data: %s", e)
            macro_data = {}

    if market_data is None:
        try:
            from takata_engine.agents.premarket_briefing import _gather_market_data
            market_data = _gather_market_data()
        except Exception as e:
            logger.warning("Could not gather market data: %s", e)
            market_data = {}

    import pytz
    brt_now = datetime.now(pytz.timezone("America/Sao_Paulo"))
    brt_hour = brt_now.hour

    # Check if in PTAX window
    ptax_warning = ""
    if brt_hour in (10, 11):
        ptax_warning = "PTAX WINDOW ACTIVE (10-11 BRT) — BCB dealer consultation in progress. Institutional flow dominates. Avoid new WDO positions."
    elif brt_hour in (13, 14):
        ptax_warning = "PTAX WINDOW ACTIVE (13-14 BRT) — Last PTAX fixing window. Expect volatility spikes. Avoid new WDO positions."

    briefing_ref = ""
    if briefing_context:
        briefing_ref = f"""
**Morning Briefing Reference:**
- Headline: {briefing_context.get('headline', 'N/A')}
- WDO bias: {briefing_context.get('wdo_bias', 'neutral')} ({briefing_context.get('wdo_bias_strength', 0):.0%})
- MES bias: {briefing_context.get('mes_bias', 'neutral')} ({briefing_context.get('mes_bias_strength', 0):.0%})
- Regime: {briefing_context.get('macro_regime', 'unknown')}
"""

    prompt = f"""Generate a macro flash report for right now.

**Current Time:** {brt_now.strftime('%H:%M BRT, %A %B %d')}
{ptax_warning}

## Macro Indicators (FRED — US)
{json.dumps(macro_data.get('fred', {}), indent=2, default=str)[:2000]}

## Macro Indicators (BCB — Brazil)
{json.dumps(macro_data.get('bcb', {}), indent=2, default=str)[:2000]}

## Market Data
{json.dumps(market_data, indent=2, default=str)[:1500]}
{briefing_ref}

Respond with ONLY the JSON."""

    result = call_claude(
        system=SYSTEM_PROMPT,
        user_prompt=prompt,
        model=MODEL_FAST,
        max_tokens=800,
    )

    if isinstance(result, dict) and not result.get("_parse_error"):
        result["generated_at"] = brt_now.isoformat()
        result["brt_hour"] = brt_hour
        result["ptax_active"] = brt_hour in (10, 11, 13, 14)
        return result

    return {
        "headline": "Macro flash unavailable",
        "flash_report": result.get("raw_text", str(result))[:500] if isinstance(result, dict) else str(result)[:500],
        "urgency": "low",
        "generated_at": brt_now.isoformat(),
    }
