"""Risk Narrator agent — live risk status in natural language.

Translates dry risk metrics (P&L, drawdown, loss streak, heat index)
into actionable human narrative. Warns when conditions deteriorate.

Uses Haiku for real-time speed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from takata_core.agents.llm_client import call_claude, MODEL_FAST

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a risk management advisor for a Brazilian futures day trader.

Given the trader's current session risk metrics, generate a concise risk narrative.

The trader's limits:
- Daily loss limit: configurable (typically R$5,000-20,000)
- Max trades/day: 8
- Target: R$20,000 BRL/day with CONSISTENCY
- Instruments: WDO (R$10/pt) and MES ($5/pt)

Risk severity levels:
- GREEN: Under 30% of daily loss limit used, positive or flat P&L
- YELLOW: 30-60% of loss limit used, or 2+ consecutive losses
- ORANGE: 60-80% of loss limit used, consider stopping
- RED: >80% of loss limit used, MUST stop trading

Respond with JSON:
{
  "severity": "green" | "yellow" | "orange" | "red",
  "headline": "One-line status (e.g., 'On track — 2 winners, P&L +R$4,200')",
  "narrative": "2-3 sentences describing current risk state and recommendation",
  "should_stop": false | true,
  "sizing_advice": "Full size" | "Half size" | "Quarter size" | "Stop trading",
  "next_action": "One actionable recommendation"
}

Be direct. If the trader should stop, say so firmly. Don't sugarcoat deteriorating conditions."""


def narrate_risk(risk_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate risk narrative from current session metrics.

    Parameters
    ----------
    risk_data : dict
        Current risk state: daily_pnl, daily_pnl_brl, trades_today,
        wins, losses, consecutive_losses, drawdown_pct, heat_index,
        max_daily_loss, loss_limit_used_pct, open_positions,
        largest_win, largest_loss.

    Returns
    -------
    dict
        severity, headline, narrative, should_stop, sizing_advice, next_action
    """
    prompt = f"""Current session risk status:

**Daily P&L:** {risk_data.get('daily_pnl_brl', 0):+,.0f} BRL
**Trades Today:** {risk_data.get('trades_today', 0)} / 8 max
**Wins:** {risk_data.get('wins', 0)} | **Losses:** {risk_data.get('losses', 0)}
**Win Rate Today:** {risk_data.get('win_rate', 0):.0%}
**Consecutive Losses:** {risk_data.get('consecutive_losses', 0)}
**Drawdown:** {risk_data.get('drawdown_pct', 0):.1%}
**Heat Index:** {risk_data.get('heat_index', 0)}/10

**Limits:**
- Daily Loss Limit: {risk_data.get('max_daily_loss', 20000):,.0f} BRL
- Loss Limit Used: {risk_data.get('loss_limit_used_pct', 0):.0%}

**Open Positions:** {risk_data.get('open_positions', 0)}
**Largest Win:** {risk_data.get('largest_win', 0):+,.0f} BRL
**Largest Loss:** {risk_data.get('largest_loss', 0):+,.0f} BRL

**Current Time (BRT):** {risk_data.get('brt_time', '??')}
**Session Phase:** {risk_data.get('session_phase', 'unknown')}

Respond with ONLY the JSON."""

    result = call_claude(
        system=SYSTEM_PROMPT,
        user_prompt=prompt,
        model=MODEL_FAST,
        max_tokens=400,
    )

    if isinstance(result, dict) and not result.get("_parse_error"):
        return result

    return {
        "severity": "yellow",
        "headline": "Risk analysis unavailable",
        "narrative": "Could not generate risk narrative. Monitor manually.",
        "should_stop": False,
        "sizing_advice": "Half size",
        "next_action": "Check risk dashboard manually",
    }
