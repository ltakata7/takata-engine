"""Shared Claude API client for all trading system agents.

Handles API key loading, model selection, and structured JSON responses.
All agents import this instead of directly using the anthropic SDK.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Model tiers — agents pick based on latency vs depth tradeoff
MODEL_FAST = "claude-haiku-4-5-20251001"      # <1s, cheap — real-time commentary
MODEL_BALANCED = "claude-sonnet-4-20250514"    # ~2s — daily briefings, analysis
MODEL_DEEP = "claude-opus-4-20250514"          # ~5s — deep research, weekly reviews

DEFAULT_MODEL = MODEL_FAST


def get_client():
    """Return an authenticated Anthropic client."""
    from dotenv import load_dotenv
    load_dotenv(override=True)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")

    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def call_claude(
    system: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1500,
    expect_json: bool = True,
) -> Dict[str, Any] | str:
    """Call Claude and return parsed JSON or raw text.

    Parameters
    ----------
    system : str
        System prompt defining the agent's role.
    user_prompt : str
        The user message with data context.
    model : str
        Model ID. Use MODEL_FAST for real-time, MODEL_BALANCED for daily,
        MODEL_DEEP for weekly/research tasks.
    max_tokens : int
        Max response tokens.
    expect_json : bool
        If True, parse response as JSON. Falls back to raw text on parse error.

    Returns
    -------
    dict or str
        Parsed JSON dict if expect_json=True, otherwise raw text.
    """
    client = get_client()

    t0 = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    elapsed = time.time() - t0
    logger.info("Claude API [%s] responded in %.1fs (%d tokens)", model, elapsed, response.usage.output_tokens)

    text = response.content[0].text.strip()

    if not expect_json:
        return text

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Claude response as JSON, returning raw text")
        return {"raw_text": text, "_parse_error": True}
