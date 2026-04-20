"""Composite risk-on / risk-off regime classifier for WDO fast-timeframe trading.

Combines short-term momentum of global risk proxies (ES, NQ, BIT) with
Brazil-specific instruments (IND, VXBR) into a single regime score.
The regime is a multiplier on the expected WDO direction: in strong
risk-off regimes, WDO signals long have extra weight; in strong risk-on,
WDO shorts do.

**Why composite risk regime matters for WDO 3-5min trading:**
- Raw signals can be high-quality but wrong-regime — a WDO short with
  strong delta divergence is a low-hit-rate trade when global risk
  is clearly risk-off (money fleeing into USD)
- Regime scoring lets us *weight* local signals by global context
  instead of gating them (don't veto good setups, just size them down)
- The regime itself changes — transitions mark the start of trend days;
  regime stability marks chop days

Regime categories:
- ``strong_risk_on``  — score <= -60: ES/NQ/IND rising together, VXBR down,
                        WDO typically falling or flat
- ``risk_on``          — score -60 to -20
- ``neutral``          — score -20 to +20
- ``risk_off``         — score +20 to +60
- ``strong_risk_off``  — score >= +60: ES/NQ/IND selling, VXBR up,
                        WDO typically rising (flight to USD)

Signal types generated:
- ``risk_on_strong``        — composite score <= -60
- ``risk_on``                — -60 < score <= -20
- ``risk_off_strong``       — composite score >= +60
- ``risk_off``               — +20 <= score < +60
- ``regime_transition_bull`` — regime just shifted risk-off → risk-on
- ``regime_transition_bear`` — regime just shifted risk-on → risk-off
- ``regime_stable_chop``    — score in neutral band for 5+ min

Call ``update(prices)`` with {IND, ES, NQ, BIT, VXBR} prices every cycle.
Call ``signal()`` for the regime dict.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import pytz

logger = logging.getLogger(__name__)

BRT = pytz.timezone("America/Sao_Paulo")

# Rolling price buffer per instrument (5 min @ 2s tick = 150)
PRICE_BUFFER = 150

# Momentum lookback for regime score (in ticks ≈ 1 min)
MOMENTUM_LOOKBACK = 30

# Neutral band boundaries
NEUTRAL_LOW = -20.0
NEUTRAL_HIGH = 20.0
STRONG_LOW = -60.0
STRONG_HIGH = 60.0

# How long regime must stay flat to count as "stable chop"
STABILITY_WINDOW = 150  # ≈ 5 min


@dataclass
class RegimeState:
    """Full regime-tracker state."""
    date: str = ""
    prices: Dict[str, Deque[float]] = field(default_factory=dict)
    # History of composite scores for transition detection
    score_history: Deque[Tuple[float, float, str]] = field(
        default_factory=lambda: deque(maxlen=STABILITY_WINDOW + 30)
    )  # (ts, score, regime)

    def reset(self, today: str) -> None:
        self.date = today
        self.prices = {
            "IND": deque(maxlen=PRICE_BUFFER),
            "ES": deque(maxlen=PRICE_BUFFER),
            "NQ": deque(maxlen=PRICE_BUFFER),
            "BIT": deque(maxlen=PRICE_BUFFER),
            "VXBR": deque(maxlen=PRICE_BUFFER),
        }
        self.score_history = deque(maxlen=STABILITY_WINDOW + 30)


def _pct_change(prices: List[float], lookback: int) -> float:
    """Percent change over the last `lookback` samples (0 if not enough)."""
    if len(prices) < lookback + 1:
        return 0.0
    ref = prices[-lookback - 1]
    cur = prices[-1]
    if ref <= 0:
        return 0.0
    return (cur - ref) / ref * 100.0


def _classify(score: float) -> str:
    """Map a composite score to a regime label."""
    if score <= STRONG_LOW:
        return "strong_risk_on"
    elif score <= NEUTRAL_LOW:
        return "risk_on"
    elif score < NEUTRAL_HIGH:
        return "neutral"
    elif score < STRONG_HIGH:
        return "risk_off"
    else:
        return "strong_risk_off"


class RiskRegimeTracker:
    """Composite risk-on/risk-off regime classifier."""

    def __init__(self) -> None:
        self._state = RegimeState()
        self._state.reset(datetime.now(BRT).strftime("%Y-%m-%d"))

    def update(self, prices: Dict[str, float]) -> None:
        """Record latest prices for tracked instruments."""
        now = datetime.now(BRT)
        today_str = now.strftime("%Y-%m-%d")
        if self._state.date != today_str:
            self._state.reset(today_str)

        for sym, buf in self._state.prices.items():
            p = prices.get(sym, 0)
            if p and p > 0:
                buf.append(float(p))

    def _compute_score(self) -> Tuple[float, Dict[str, float]]:
        """Compute composite score and return breakdown.

        Score convention:
        - Positive = risk-off (global equities down, VXBR up)
        - Negative = risk-on

        Weights (tuned for WDO relevance):
        - ES: 30%   (US primary risk benchmark)
        - NQ: 15%   (US tech risk)
        - IND: 25%  (Brazil local risk; heavier weight in BR hours)
        - BIT: 10%  (global speculation proxy)
        - VXBR: 20% (Brazil fear index; inverted contribution)
        """
        breakdown: Dict[str, float] = {}
        weights = {"ES": 30, "NQ": 15, "IND": 25, "BIT": 10, "VXBR": 20}

        # Equity momentum: negative % change contributes positively (risk-off)
        es_pct = _pct_change(list(self._state.prices["ES"]), MOMENTUM_LOOKBACK)
        nq_pct = _pct_change(list(self._state.prices["NQ"]), MOMENTUM_LOOKBACK)
        ind_pct = _pct_change(list(self._state.prices["IND"]), MOMENTUM_LOOKBACK)
        bit_pct = _pct_change(list(self._state.prices["BIT"]), MOMENTUM_LOOKBACK)
        # For VXBR: positive % change = more fear = risk-off (adds to score)
        vxbr_pct = _pct_change(list(self._state.prices["VXBR"]), MOMENTUM_LOOKBACK)

        # Scale each % change to a -100..+100 contribution (cap at ±1% move)
        def scale(pct: float, cap: float = 1.0) -> float:
            return max(-100.0, min(100.0, -pct / cap * 100.0))  # negative = risk-on

        breakdown["ES"] = scale(es_pct) * weights["ES"] / 100
        breakdown["NQ"] = scale(nq_pct) * weights["NQ"] / 100
        breakdown["IND"] = scale(ind_pct) * weights["IND"] / 100
        breakdown["BIT"] = scale(bit_pct) * weights["BIT"] / 100
        # VXBR: inverted so rising VXBR = positive score
        breakdown["VXBR"] = -scale(vxbr_pct) * weights["VXBR"] / 100

        score = sum(breakdown.values())
        return score, breakdown

    def signal(self) -> Dict[str, Any]:
        """Generate regime signal dict."""
        result: Dict[str, Any] = {
            "score": 0.0,
            "regime": "insufficient_data",
            "breakdown": {},
            "pct_changes": {},
            "signal": "neutral",
            "score_adjustment": 0.0,
            "reasons": [],
        }
        reasons: List[str] = result["reasons"]
        adj = 0.0

        # Need at least 2 instruments with data
        with_data = sum(1 for b in self._state.prices.values() if len(b) >= MOMENTUM_LOOKBACK + 1)
        if with_data < 2:
            return result

        score, breakdown = self._compute_score()
        regime = _classify(score)
        result["score"] = round(score, 1)
        result["regime"] = regime
        result["breakdown"] = {k: round(v, 2) for k, v in breakdown.items()}
        result["pct_changes"] = {
            sym: round(_pct_change(list(buf), MOMENTUM_LOOKBACK), 3)
            for sym, buf in self._state.prices.items()
        }

        # Record score history for transition detection
        now_ts = datetime.now(BRT).timestamp()
        self._state.score_history.append((now_ts, score, regime))

        # ── Regime signals ──
        if regime == "strong_risk_on":
            reasons.append("risk_on_strong")
            adj += 0.06
            result["signal"] = "risk_on_strong"
        elif regime == "risk_on":
            reasons.append("risk_on")
            adj += 0.03
            result["signal"] = "risk_on"
        elif regime == "strong_risk_off":
            reasons.append("risk_off_strong")
            adj += 0.06
            result["signal"] = "risk_off_strong"
        elif regime == "risk_off":
            reasons.append("risk_off")
            adj += 0.03
            result["signal"] = "risk_off"

        # ── Regime transition detection ──
        history = list(self._state.score_history)
        if len(history) >= 30:
            # Compare current regime to regime 30 ticks (≈1 min) ago
            prior_regime = history[-30][2]
            prior_is_on = prior_regime in ("strong_risk_on", "risk_on")
            prior_is_off = prior_regime in ("strong_risk_off", "risk_off")
            cur_is_on = regime in ("strong_risk_on", "risk_on")
            cur_is_off = regime in ("strong_risk_off", "risk_off")

            if prior_is_off and cur_is_on:
                reasons.append("regime_transition_bull")
                adj += 0.05
                result["signal"] = "regime_flip_to_on"
            elif prior_is_on and cur_is_off:
                reasons.append("regime_transition_bear")
                adj += 0.05
                result["signal"] = "regime_flip_to_off"

        # ── Stability / chop detection ──
        if len(history) >= STABILITY_WINDOW:
            last_window = history[-STABILITY_WINDOW:]
            regimes_in_window = {r for _, _, r in last_window}
            if regimes_in_window == {"neutral"}:
                reasons.append("regime_stable_chop")
                # Chop regime reduces conviction slightly
                adj -= 0.02

        result["score_adjustment"] = round(adj, 3)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Dump state for API/frontend."""
        return {
            "date": self._state.date,
            "samples": {sym: len(buf) for sym, buf in self._state.prices.items()},
            "history_depth": len(self._state.score_history),
        }


# Singleton accessor
_regime_tracker_instance: Optional[RiskRegimeTracker] = None


def get_risk_regime_tracker() -> RiskRegimeTracker:
    """Get the singleton risk-regime tracker."""
    global _regime_tracker_instance
    if _regime_tracker_instance is None:
        _regime_tracker_instance = RiskRegimeTracker()
    return _regime_tracker_instance
