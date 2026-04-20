"""Signal Explainer agent — generates human-readable signal commentary.

When a WDO/MES signal fires, this agent explains WHY in 1-3 sentences
so the trader understands the confluence driving the setup.

Uses Haiku for speed (<1s) since this runs in the live trading loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from takata_core.agents.llm_client import call_claude, MODEL_FAST

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a concise trading signal analyst for a Brazilian futures day trader.

When given a signal with its indicators and reasons, explain in 1-3 SHORT sentences:
1. What confluence of factors triggered this signal
2. The key risk to watch
3. One actionable note (e.g., "wait for pullback to VWAP" or "trail stop at 5440")

The trader trades WDO (Mini Dollar B3, R$10/point) and MES (Micro S&P, $5/point).
PTAX windows (10-11, 13-14 BRT) are dangerous — if the signal is near one, warn explicitly.

Historical context:
- SHORT WDO works best at 12:00, 15:00, 17:00 BRT
- LONG WDO only works with strong buyer aggression (agg > 2000)
- The trader targets R$20,000/day with CONSISTENCY

Respond with JSON:
{
  "explanation": "1-3 sentence explanation of why this signal fired",
  "conviction": "high" | "medium" | "low",
  "key_risk": "One sentence about the main risk",
  "action_note": "One actionable tip"
}

Be direct. No hedge words. State conviction clearly."""


def explain_signal(signal_data: Dict[str, Any]) -> Dict[str, str]:
    """Generate a human-readable explanation for a trading signal.

    Parameters
    ----------
    signal_data : dict
        Signal data including: instrument, direction, strength, score,
        reasons, indicators, regime, session_phase, bias, etc.

    Returns
    -------
    dict
        explanation, conviction, key_risk, action_note
    """
    prompt = f"""Explain this trading signal:

**Instrument:** {signal_data.get('instrument', 'unknown')}
**Direction:** {signal_data.get('direction', 'none')}
**Strength:** {signal_data.get('strength', 0):.2f}
**Score:** {signal_data.get('score', 0):.2f}

**Reasons (confluence factors):**
{json.dumps(signal_data.get('reasons', []), indent=2)}

**Indicators:**
{json.dumps(signal_data.get('indicators', {}), indent=2, default=str)}

**Regime:** {signal_data.get('regime', 'unknown')}
**Session Phase:** {signal_data.get('session_phase', 'unknown')}
**MTF Bias:** {signal_data.get('bias', 'neutral')}
**Briefing Bias:** {signal_data.get('briefing_bias', 'none')}

**Current BRT Hour:** {signal_data.get('brt_hour', '??')}

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
        "explanation": result.get("raw_text", str(result))[:300],
        "conviction": "low",
        "key_risk": "Could not generate analysis",
        "action_note": "Use your own judgment",
    }
