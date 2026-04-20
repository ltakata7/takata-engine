"""Session Debrief agent — end-of-day and weekly performance reviews.

Generates coaching-style performance analysis from trade history.
Identifies patterns, recurring mistakes, and areas of strength.

Uses Sonnet for depth — this runs after market close, latency is fine.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from takata_core.agents.llm_client import call_claude, MODEL_BALANCED

logger = logging.getLogger(__name__)

DEBRIEF_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "debriefs"

DAILY_SYSTEM_PROMPT = """You are a senior trading performance coach reviewing a day trader's session.

The trader day-trades:
- WDO (Mini Dollar B3) — R$10/point, up to 100 contracts
- MES (Micro E-mini S&P) — $5/point, 2-3 contracts
- Target: R$20,000 BRL/day with CONSISTENCY

Rules they should follow:
- Avoid PTAX windows (10-11, 13-14 BRT)
- SHORT WDO best at 12:00, 15:00, 17:00 BRT
- LONG WDO only with strong buyer aggression
- Max 8 trades/day, focus on 2-5 high-conviction setups
- Consistency > upside

Respond with JSON:
{
  "headline": "One-line summary of the session",
  "grade": "A+" | "A" | "B" | "C" | "D" | "F",
  "pnl_assessment": "Were they profitable? By how much?",
  "best_trade": "Which trade was best executed and why",
  "worst_trade": "Which trade was worst executed and why",
  "discipline_score": 0-10,
  "discipline_notes": "Were rules followed? Overtrading? PTAX violations?",
  "patterns_noticed": ["pattern1", "pattern2"],
  "tomorrow_focus": "One thing to focus on tomorrow",
  "emotional_read": "Assessment of trader's emotional state based on trade patterns"
}

Be honest, direct, and constructive. Celebrate discipline even on losing days."""


WEEKLY_SYSTEM_PROMPT = """You are a senior trading performance coach reviewing a day trader's WEEKLY performance.

The trader day-trades WDO (Mini Dollar B3, R$10/point) and MES (Micro S&P, $5/point).
Target: R$20,000 BRL/day = R$100,000/week with CONSISTENCY.

Analyze the full week's trades and provide strategic coaching.

Respond with JSON:
{
  "headline": "One-line week summary",
  "weekly_grade": "A+" | "A" | "B" | "C" | "D" | "F",
  "total_pnl_brl": 0,
  "daily_breakdown": [{"day": "Mon", "pnl": 0, "trades": 0, "grade": "B"}],
  "win_rate": 0.0,
  "best_day": "Which day and why",
  "worst_day": "Which day and why",
  "recurring_patterns": ["pattern1", "pattern2"],
  "strength": "Trader's biggest strength this week",
  "weakness": "Trader's biggest area for improvement",
  "strategy_adjustment": "One strategic change for next week",
  "consistency_score": 0-10,
  "edge_assessment": "Is the trader's edge growing, stable, or eroding?"
}

Think like a portfolio manager reviewing a junior trader's book."""


def generate_daily_debrief(trades: List[Dict[str, Any]], session_date: str = None) -> Dict[str, Any]:
    """Generate end-of-day performance debrief.

    Parameters
    ----------
    trades : list
        All trades from today's session. Each dict should have:
        instrument, side, entry_price, exit_price, size, pnl, pnl_brl,
        entry_time, exit_time, signal_strength, signal_reasons, regime.
    session_date : str, optional
        Date string (YYYY-MM-DD). Defaults to today.

    Returns
    -------
    dict
        Full debrief analysis.
    """
    if not session_date:
        session_date = date.today().isoformat()

    if not trades:
        return {
            "headline": "No trades today",
            "grade": "N/A",
            "pnl_assessment": "No trades recorded",
            "discipline_score": 10,
            "discipline_notes": "No trades is sometimes the best trade",
            "tomorrow_focus": "Look for high-conviction setups only",
        }

    total_pnl = sum(t.get("pnl_brl", t.get("pnl", 0)) for t in trades)
    wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
    losses = len(trades) - wins

    trades_summary = []
    for i, t in enumerate(trades, 1):
        trades_summary.append({
            "trade_num": i,
            "instrument": t.get("instrument"),
            "side": t.get("side"),
            "entry": t.get("entry_price"),
            "exit": t.get("exit_price"),
            "size": t.get("size"),
            "pnl": t.get("pnl"),
            "pnl_brl": t.get("pnl_brl"),
            "entry_time": t.get("entry_time"),
            "exit_time": t.get("exit_time"),
            "signal_strength": t.get("signal_strength"),
            "reasons": t.get("signal_reasons", []),
        })

    prompt = f"""Review today's trading session ({session_date}):

**Summary:**
- Total trades: {len(trades)}
- Wins: {wins} | Losses: {losses} | Win rate: {wins/len(trades):.0%}
- Total P&L: {total_pnl:+,.0f} BRL
- Target: R$20,000/day

**All Trades:**
{json.dumps(trades_summary, indent=2, default=str)}

Respond with ONLY the JSON."""

    result = call_claude(
        system=DAILY_SYSTEM_PROMPT,
        user_prompt=prompt,
        model=MODEL_BALANCED,
        max_tokens=1000,
    )

    if isinstance(result, dict) and not result.get("_parse_error"):
        result["date"] = session_date
        result["total_pnl_brl"] = total_pnl
        result["trades_count"] = len(trades)
        result["win_rate"] = wins / len(trades) if trades else 0
        _save_debrief(session_date, "daily", result)
        return result

    fallback = {
        "headline": f"Session: {len(trades)} trades, {total_pnl:+,.0f} BRL",
        "grade": "?",
        "date": session_date,
        "total_pnl_brl": total_pnl,
        "trades_count": len(trades),
    }
    _save_debrief(session_date, "daily", fallback)
    return fallback


def generate_weekly_debrief(daily_trades: Dict[str, List[Dict]], week_label: str = None) -> Dict[str, Any]:
    """Generate weekly performance review.

    Parameters
    ----------
    daily_trades : dict
        Keys are date strings, values are lists of trades for that day.
    week_label : str, optional
        Label like "2026-W15". Auto-generated if not provided.

    Returns
    -------
    dict
        Weekly review analysis.
    """
    if not week_label:
        week_label = f"Week of {min(daily_trades.keys()) if daily_trades else date.today().isoformat()}"

    daily_summaries = []
    all_trades = []
    for d in sorted(daily_trades.keys()):
        day_trades = daily_trades[d]
        all_trades.extend(day_trades)
        pnl = sum(t.get("pnl_brl", t.get("pnl", 0)) for t in day_trades)
        wins = sum(1 for t in day_trades if (t.get("pnl") or 0) > 0)
        daily_summaries.append({
            "date": d,
            "trades": len(day_trades),
            "wins": wins,
            "losses": len(day_trades) - wins,
            "pnl_brl": round(pnl, 2),
        })

    total_pnl = sum(d["pnl_brl"] for d in daily_summaries)
    total_trades = sum(d["trades"] for d in daily_summaries)
    total_wins = sum(d["wins"] for d in daily_summaries)

    prompt = f"""Review this week's trading performance ({week_label}):

**Weekly Summary:**
- Trading days: {len(daily_summaries)}
- Total trades: {total_trades}
- Total wins: {total_wins} | Losses: {total_trades - total_wins}
- Win rate: {total_wins/total_trades:.0%} if total_trades else 'N/A'
- Total P&L: {total_pnl:+,.0f} BRL (target: R$100,000/week)

**Daily Breakdown:**
{json.dumps(daily_summaries, indent=2)}

**All Trades (chronological):**
{json.dumps([{
    'date': t.get('date', '?'),
    'instrument': t.get('instrument'),
    'side': t.get('side'),
    'pnl': t.get('pnl'),
    'pnl_brl': t.get('pnl_brl'),
    'signal_strength': t.get('signal_strength'),
    'entry_time': t.get('entry_time'),
} for t in all_trades], indent=2, default=str)}

Respond with ONLY the JSON."""

    result = call_claude(
        system=WEEKLY_SYSTEM_PROMPT,
        user_prompt=prompt,
        model=MODEL_BALANCED,
        max_tokens=1500,
    )

    if isinstance(result, dict) and not result.get("_parse_error"):
        result["week"] = week_label
        _save_debrief(week_label.replace(" ", "_"), "weekly", result)
        return result

    return {
        "headline": f"Week: {total_trades} trades, {total_pnl:+,.0f} BRL",
        "weekly_grade": "?",
        "week": week_label,
        "total_pnl_brl": total_pnl,
    }


def _save_debrief(label: str, kind: str, data: dict):
    """Save debrief to disk."""
    DEBRIEF_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBRIEF_DIR / f"{kind}_{label}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    logger.info("Debrief saved: %s", path)
