"""Trade Analyst agent — post-trade analysis and learning.

After a trade is closed, this agent analyzes WHY it won or lost,
identifies patterns, and generates actionable lessons.

Uses Haiku for speed since traders want quick feedback after closing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from takata_core.agents.llm_client import call_claude, MODEL_FAST

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a trading performance coach for a Brazilian futures day trader.

When given a closed trade with its entry, exit, P&L, and signal context, provide:
1. A clear verdict: was the trade well-executed or not?
2. What went right and what went wrong
3. One specific, actionable lesson for next time

The trader day-trades:
- WDO (Mini Dollar B3) — R$10/point, targets R$20k/day
- MES (Micro E-mini S&P) — $5/point, 2-3 contracts

Key rules the trader should follow:
- Avoid PTAX windows (10-11, 13-14 BRT) — 0% historical win rate
- SHORT WDO best at 12:00, 15:00, 17:00 BRT
- LONG WDO only with strong buyer aggression (agg > 2000)
- Maximum 8 trades/day — focus on 2-5 high-conviction setups
- Consistency over big swings

Respond with JSON:
{
  "verdict": "good_trade" | "bad_trade" | "unlucky" | "lucky",
  "grade": "A" | "B" | "C" | "D" | "F",
  "what_went_right": "1 sentence",
  "what_went_wrong": "1 sentence or 'Nothing — clean execution'",
  "lesson": "1 specific actionable lesson",
  "pattern_note": "Any recurring pattern you notice (optional)",
  "emotional_check": "Was this likely a revenge trade, FOMO, or disciplined entry?"
}

Be honest and direct. Don't sugarcoat losses. Don't over-praise wins."""


def analyze_trade(trade: Dict[str, Any], recent_trades: List[Dict] = None) -> Dict[str, str]:
    """Analyze a closed trade and generate coaching feedback.

    Parameters
    ----------
    trade : dict
        Closed trade data: instrument, side, entry_price, exit_price, size,
        pnl, pnl_brl, fees, signal_strength, signal_reasons, regime,
        entry_time, exit_time, duration_minutes.
    recent_trades : list, optional
        Last 5-10 trades for pattern detection (streaks, revenge trading).

    Returns
    -------
    dict
        verdict, grade, what_went_right, what_went_wrong, lesson, pattern_note
    """
    # Build context about recent performance
    recent_context = ""
    if recent_trades:
        wins = sum(1 for t in recent_trades if (t.get("pnl") or 0) > 0)
        losses = len(recent_trades) - wins
        streak = 0
        for t in reversed(recent_trades):
            if (t.get("pnl") or 0) > 0:
                if streak >= 0:
                    streak += 1
                else:
                    break
            else:
                if streak <= 0:
                    streak -= 1
                else:
                    break
        recent_context = f"""
**Recent Performance (last {len(recent_trades)} trades):**
- Wins: {wins}, Losses: {losses}
- Current streak: {streak:+d} ({'winning' if streak > 0 else 'losing'})
- Recent P&L: {sum(t.get('pnl', 0) for t in recent_trades):.2f}
"""

    prompt = f"""Analyze this closed trade:

**Instrument:** {trade.get('instrument', 'unknown')}
**Side:** {trade.get('side', 'unknown')}
**Entry:** {trade.get('entry_price', 0)} at {trade.get('entry_time', '??')}
**Exit:** {trade.get('exit_price', 0)} at {trade.get('exit_time', '??')}
**Size:** {trade.get('size', 0)} contracts
**P&L:** {trade.get('pnl', 0):.2f} points ({trade.get('pnl_brl', 'N/A')} BRL)
**Fees:** {trade.get('fees', 0):.2f}
**Duration:** {trade.get('duration_minutes', '??')} minutes

**Signal Context:**
- Strength: {trade.get('signal_strength', 'N/A')}
- Reasons: {json.dumps(trade.get('signal_reasons', []))}
- Regime: {trade.get('regime', 'unknown')}
- Session phase: {trade.get('session_phase', 'unknown')}
{recent_context}

Respond with ONLY the JSON."""

    result = call_claude(
        system=SYSTEM_PROMPT,
        user_prompt=prompt,
        model=MODEL_FAST,
        max_tokens=500,
    )

    if isinstance(result, dict) and not result.get("_parse_error"):
        return result

    return {
        "verdict": "unknown",
        "grade": "?",
        "what_went_right": "Analysis unavailable",
        "what_went_wrong": "Analysis unavailable",
        "lesson": "Review trade manually",
        "pattern_note": "",
        "emotional_check": "unknown",
    }
