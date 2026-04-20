"""Pre-market briefing agent — daily macro thesis via Claude API.

Gathers overnight data from FRED, BCB, DI curve, market emails,
and generates a structured daily trading thesis for WDO and MES.

Designed to run at 08:30 BRT before market open (09:00).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BRIEFING_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "briefings"


@dataclass
class DailyThesis:
    """Structured output from the pre-market briefing agent."""

    date: str
    generated_at: str

    # Macro context
    macro_regime: str = ""              # goldilocks / reflation / stagflation / deflation / transition
    macro_confidence: float = 0.0

    # Instrument biases
    wdo_bias: str = "neutral"           # long / short / neutral
    wdo_bias_strength: float = 0.0      # 0-1
    wdo_key_levels: List[float] = field(default_factory=list)
    wdo_thesis: str = ""

    mes_bias: str = "neutral"
    mes_bias_strength: float = 0.0
    mes_key_levels: List[float] = field(default_factory=list)
    mes_thesis: str = ""

    # Risk alerts
    risk_events: List[str] = field(default_factory=list)
    ptax_expectation: str = ""
    volatility_outlook: str = ""        # low / normal / elevated / extreme

    # Summary
    headline: str = ""
    full_analysis: str = ""

    # Data sources used
    sources: List[str] = field(default_factory=list)
    email_summaries: List[Dict[str, str]] = field(default_factory=list)

    def save(self) -> Path:
        BRIEFING_DIR.mkdir(parents=True, exist_ok=True)
        path = BRIEFING_DIR / f"{self.date}.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
        logger.info("Briefing saved: %s", path)
        return path

    @classmethod
    def load(cls, d: str) -> Optional["DailyThesis"]:
        path = BRIEFING_DIR / f"{d}.json"
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text()))

    @classmethod
    def load_latest(cls) -> Optional["DailyThesis"]:
        BRIEFING_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(BRIEFING_DIR.glob("*.json"), reverse=True)
        if not files:
            return None
        return cls(**json.loads(files[0].read_text()))


def _gather_macro_data() -> Dict[str, Any]:
    """Pull latest macro indicators from FRED + BCB feeds."""
    context = {"fred": {}, "bcb": {}, "di_curve": {}, "macro_cycle": {}}

    # FRED data
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        from takata_engine.macro.data_feeds.fred_feed import FREDFeed
        fred = FREDFeed()
        bundle = fred.fetch_all()
        for ind in bundle.indicators:
            context["fred"][ind.name] = {
                "value": ind.latest_value,
                "change_pct": ind.change_pct,
                "interpretation": ind.interpretation,
                "signal": ind.signal,
                "date": ind.date,
            }
        logger.info("FRED: %d indicators loaded", len(bundle.indicators))
    except Exception as e:
        logger.warning("FRED fetch failed: %s", e)

    # BCB data
    try:
        from takata_engine.macro.data_feeds.bcb_feed import BCBFeed
        bcb = BCBFeed()
        bundle = bcb.fetch_all()
        for ind in bundle.indicators:
            context["bcb"][ind.name] = {
                "value": ind.latest_value,
                "change_pct": ind.change_pct,
                "interpretation": ind.interpretation,
                "signal": ind.signal,
                "date": ind.date,
            }
        logger.info("BCB: %d indicators loaded", len(bundle.indicators))
    except Exception as e:
        logger.warning("BCB fetch failed: %s", e)

    # Macro cycle classifier
    try:
        from api.routers.macro import get_macro_cycle
        cycle = get_macro_cycle()
        if cycle:
            context["macro_cycle"] = cycle
    except Exception as e:
        logger.debug("Macro cycle fetch failed: %s", e)

    return context


def _gather_market_data() -> Dict[str, Any]:
    """Pull live/recent market data from bridge and IBKR."""
    context = {"wdo": {}, "mes": {}, "di_curve": {}}

    # WDO from bridge
    try:
        import requests
        bridge = os.getenv("WDO_BRIDGE_URL", "http://127.0.0.1:8002")
        r = requests.get(f"{bridge}/dashboard", timeout=5)
        data = r.json()
        context["wdo"]["dashboard"] = {
            "instruments": data.get("instruments", []),
        }
        # DI curve
        r2 = requests.get(f"{bridge}/quote/DI1FUT", timeout=3)
        context["di_curve"]["front"] = r2.json()
    except Exception as e:
        logger.debug("Bridge data unavailable: %s", e)

    return context


def _gather_email_context(email_summaries: List[Dict[str, str]] = None) -> str:
    """Format email summaries for the LLM prompt.

    Email summaries are passed in from the caller (API endpoint or scheduled task)
    which handles Gmail API access.
    """
    if not email_summaries:
        return ""

    lines = ["## Market Emails (overnight)\n"]
    for em in email_summaries[:10]:
        lines.append(f"**From:** {em.get('from', 'unknown')}")
        lines.append(f"**Subject:** {em.get('subject', '')}")
        lines.append(f"**Snippet:** {em.get('snippet', '')}")
        lines.append("")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a senior macro strategist and day trading analyst for a Brazilian futures trader.

Your job is to generate a PRE-MARKET BRIEFING every morning before the B3 session opens at 09:00 BRT.

The trader day-trades:
- **WDO** (Mini Dollar futures on B3) — R$10/point, up to 100 contracts
- **MES** (Micro E-mini S&P 500 on CME) — $5/point, 2-3 contracts

Their goal is R$20,000 BRL/day with CONSISTENCY — steady profits over big swings.

PTAX formation windows (10:00, 11:00, 12:00, 13:00 BRT) are critical for WDO — BCB consults dealers during these windows and institutional flow dominates.

Historical analysis shows:
- SHORT WDO signals perform best at 12:00, 15:00, 17:00 BRT
- LONG WDO only works with strong buyer aggression (agg > 2000)
- PTAX windows (10-11, 13-14) have 0% signal win rate — avoid trading
- The trader should focus on 2-5 high-conviction trades, NOT 30+ scalps

You must output valid JSON matching this exact schema:
{
  "headline": "One-line summary of today's thesis",
  "wdo_bias": "long" | "short" | "neutral",
  "wdo_bias_strength": 0.0-1.0,
  "wdo_thesis": "2-3 sentences explaining WDO thesis for today",
  "wdo_key_levels": [level1, level2, ...],
  "mes_bias": "long" | "short" | "neutral",
  "mes_bias_strength": 0.0-1.0,
  "mes_thesis": "2-3 sentences explaining MES thesis for today",
  "mes_key_levels": [level1, level2, ...],
  "risk_events": ["event1", "event2"],
  "ptax_expectation": "What to expect from PTAX formation today",
  "volatility_outlook": "low" | "normal" | "elevated" | "extreme",
  "macro_regime": "goldilocks" | "reflation" | "stagflation" | "deflation" | "transition",
  "macro_confidence": 0.0-1.0,
  "full_analysis": "Detailed 3-5 paragraph analysis covering macro, technicals, and actionable plan"
}

Be direct and actionable. No hedging language. State your conviction clearly."""


def generate_briefing(
    email_summaries: List[Dict[str, str]] = None,
    model: str = "claude-sonnet-4-20250514",
) -> DailyThesis:
    """Generate today's pre-market briefing via Claude API.

    Parameters
    ----------
    email_summaries
        List of dicts with 'from', 'subject', 'snippet' from Gmail.
    model
        Claude model to use. Defaults to Sonnet for speed/cost.
    """
    from dotenv import load_dotenv
    load_dotenv(override=True)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Gather all context
    t0 = time.time()
    macro = _gather_macro_data()
    market = _gather_market_data()
    emails_text = _gather_email_context(email_summaries)
    gather_time = time.time() - t0
    logger.info("Data gathered in %.1fs", gather_time)

    # Build the user prompt
    today = date.today().isoformat()
    user_prompt = f"""Generate the pre-market briefing for **{today}**.

## Macro Data (FRED — US)
{json.dumps(macro.get('fred', {}), indent=2, default=str)}

## Macro Data (BCB — Brazil)
{json.dumps(macro.get('bcb', {}), indent=2, default=str)}

## Macro Cycle Classifier
{json.dumps(macro.get('macro_cycle', {}), indent=2, default=str)}

## Market Data
{json.dumps(market, indent=2, default=str)}

{emails_text}

Respond with ONLY the JSON object, no markdown fences."""

    # Call Claude
    t1 = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    api_time = time.time() - t1
    logger.info("Claude API responded in %.1fs (model=%s)", api_time, model)

    # Parse response
    text = response.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Claude response as JSON: %s\n%s", e, text[:500])
        result = {
            "headline": "Briefing generation failed — could not parse LLM response",
            "wdo_bias": "neutral", "wdo_bias_strength": 0.0, "wdo_thesis": text[:200],
            "mes_bias": "neutral", "mes_bias_strength": 0.0, "mes_thesis": "",
            "risk_events": ["LLM parse error"], "volatility_outlook": "normal",
            "macro_regime": "transition", "macro_confidence": 0.0,
            "full_analysis": text,
        }

    # Build thesis
    sources = []
    if macro.get("fred"):
        sources.append(f"FRED ({len(macro['fred'])} indicators)")
    if macro.get("bcb"):
        sources.append(f"BCB ({len(macro['bcb'])} indicators)")
    if macro.get("macro_cycle"):
        sources.append("Macro cycle classifier")
    if market.get("wdo", {}).get("dashboard"):
        sources.append("WDO bridge")
    if email_summaries:
        sources.append(f"Gmail ({len(email_summaries)} emails)")

    thesis = DailyThesis(
        date=today,
        generated_at=datetime.now().isoformat(),
        headline=result.get("headline", ""),
        macro_regime=result.get("macro_regime", "transition"),
        macro_confidence=float(result.get("macro_confidence", 0)),
        wdo_bias=result.get("wdo_bias", "neutral"),
        wdo_bias_strength=float(result.get("wdo_bias_strength", 0)),
        wdo_key_levels=result.get("wdo_key_levels", []),
        wdo_thesis=result.get("wdo_thesis", ""),
        mes_bias=result.get("mes_bias", "neutral"),
        mes_bias_strength=float(result.get("mes_bias_strength", 0)),
        mes_key_levels=result.get("mes_key_levels", []),
        mes_thesis=result.get("mes_thesis", ""),
        risk_events=result.get("risk_events", []),
        ptax_expectation=result.get("ptax_expectation", ""),
        volatility_outlook=result.get("volatility_outlook", "normal"),
        full_analysis=result.get("full_analysis", ""),
        sources=sources,
        email_summaries=email_summaries or [],
    )

    thesis.save()
    return thesis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    thesis = generate_briefing()
    print(f"\n{'='*60}")
    print(f"  {thesis.headline}")
    print(f"{'='*60}")
    print(f"  WDO: {thesis.wdo_bias.upper()} ({thesis.wdo_bias_strength:.0%})")
    print(f"  MES: {thesis.mes_bias.upper()} ({thesis.mes_bias_strength:.0%})")
    print(f"  Regime: {thesis.macro_regime} ({thesis.macro_confidence:.0%})")
    print(f"  Vol outlook: {thesis.volatility_outlook}")
    print(f"  Risk events: {', '.join(thesis.risk_events) or 'none'}")
    print(f"\n{thesis.full_analysis}")
