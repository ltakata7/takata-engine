"""Lead-lag detection across WDO and global reference instruments.

When one instrument's price move consistently precedes another's on a
short horizon, the leader becomes an anticipation signal for the follower.
In practice: IND (IBOV) often leads WDO by 20-40 seconds on Brazil risk-off
moves; ES breaks often precede WDO in US hours when the move is global.

**Why lead-lag matters for WDO 3-5min trading:**
- If IND already broke its 5-min high, WDO is likely to follow within
  ~30 seconds — you can take the WDO position BEFORE the move is obvious
- If ES rolled over 15 seconds ago, the WDO breakout you're watching is
  more likely to fail (global risk-off caught up)
- When neither leader has moved, WDO breaks are more likely to be
  idiosyncratic (PTAX-driven) — different playbook

Implementation: compute rolling correlation at several lag offsets
(leader moved K ticks before follower) and flag the offset with the
strongest absolute correlation. Report directional alignment too.

Signal types generated:
- ``leadlag_ind_bull``  — IND rising ahead of WDO rising (typical inverse)
- ``leadlag_ind_bear``  — IND falling ahead of WDO falling (weak signal —
                          expect WDO to follow down after a tiny lag)
- ``leadlag_es_bull``   — ES leading WDO upward (unusual; US USD-demand)
- ``leadlag_es_bear``   — ES leading WDO downward (risk-on caught up)
- ``leadlag_nq_signal`` — NQ leading WDO either direction
- ``leadlag_no_leader`` — nothing leads WDO clearly = idiosyncratic regime

Call ``update(wdo_price, {symbol: price})`` every scan cycle.
Call ``signal(wdo_price)`` for the lead-lag analysis dict.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)

BRT = pytz.timezone("America/Sao_Paulo")

# Instruments we test for lead-lag against WDO
LEADER_ASSETS = ("IND", "ES", "NQ")

# Price buffer length — cover enough to test lags up to 45s (≈23 ticks at 2s)
BUFFER_LEN = 120

# Lag offsets to test (in ticks; ~2s per tick)
LAG_OFFSETS = (5, 10, 15, 20, 30)  # 10s, 20s, 30s, 40s, 60s

# Minimum |correlation| to claim leadership
CORR_CONFIDENCE = 0.45

# Minimum recent move magnitude (points) to flag a signal
MIN_LEADER_MOVE = 0.5


@dataclass
class LeaderState:
    """Per-leader price buffer."""
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=BUFFER_LEN))


@dataclass
class LeadLagState:
    """Full tracker state."""
    date: str = ""
    wdo_prices: Deque[float] = field(default_factory=lambda: deque(maxlen=BUFFER_LEN))
    leaders: Dict[str, LeaderState] = field(default_factory=dict)

    def reset(self, today: str) -> None:
        self.date = today
        self.wdo_prices = deque(maxlen=BUFFER_LEN)
        self.leaders = {sym: LeaderState() for sym in LEADER_ASSETS}


def _returns(prices: List[float]) -> List[float]:
    """Convert price levels to arithmetic returns."""
    if len(prices) < 2:
        return []
    out: List[float] = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0:
            out.append((prices[i] - prices[i - 1]) / prices[i - 1])
    return out


def _lagged_corr(leader_rets: List[float], follower_rets: List[float], lag: int) -> float:
    """Correlation between leader (shifted back by `lag`) and follower.

    Positive lag = leader's returns at time t-lag correlated with
    follower's returns at time t.
    """
    if lag <= 0 or len(leader_rets) <= lag or len(follower_rets) <= lag:
        return 0.0
    x = leader_rets[:-lag]
    y = follower_rets[lag:]
    n = min(len(x), len(y))
    if n < 10:
        return 0.0
    x = x[-n:]
    y = y[-n:]
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return 0.0
    return num / math.sqrt(vx * vy)


class LeadLagDetector:
    """Detects which reference instrument leads WDO on short timescales."""

    def __init__(self) -> None:
        self._state = LeadLagState()
        self._state.reset(datetime.now(BRT).strftime("%Y-%m-%d"))

    def update(self, wdo_price: float, leader_prices: Dict[str, float]) -> None:
        """Record new prices."""
        if wdo_price <= 0:
            return
        now = datetime.now(BRT)
        today_str = now.strftime("%Y-%m-%d")
        if self._state.date != today_str:
            self._state.reset(today_str)

        self._state.wdo_prices.append(wdo_price)
        for sym in LEADER_ASSETS:
            if sym in leader_prices:
                p = leader_prices[sym]
                if p and p > 0:
                    self._state.leaders[sym].prices.append(p)

    def signal(self, wdo_price: float) -> Dict[str, Any]:
        """Generate lead-lag signal dict."""
        result: Dict[str, Any] = {
            "leaders": {},
            "best_leader": None,
            "signal": "neutral",
            "score_adjustment": 0.0,
            "reasons": [],
        }
        reasons: List[str] = result["reasons"]
        adj = 0.0

        wdo_series = list(self._state.wdo_prices)
        if len(wdo_series) < 40:
            return result

        wdo_rets = _returns(wdo_series)

        best_sym: Optional[str] = None
        best_abs_corr: float = 0.0
        best_lag: int = 0
        best_corr: float = 0.0

        for sym, ld in self._state.leaders.items():
            leader_series = list(ld.prices)
            if len(leader_series) < 40:
                continue
            # Align to shared length
            n = min(len(leader_series), len(wdo_series))
            leader_rets = _returns(leader_series[-n:])
            follower_rets = _returns(wdo_series[-n:])
            if len(leader_rets) < 30 or len(follower_rets) < 30:
                continue

            best_for_sym: Dict[str, float] = {"lag": 0, "corr": 0.0}
            for lag in LAG_OFFSETS:
                c = _lagged_corr(leader_rets, follower_rets, lag)
                if abs(c) > abs(best_for_sym["corr"]):
                    best_for_sym = {"lag": float(lag), "corr": c}

            # Recent leader move magnitude (points over last 30 ticks)
            recent_move = 0.0
            if len(leader_series) >= 30:
                recent_move = leader_series[-1] - leader_series[-30]

            result["leaders"][sym] = {
                "best_lag_ticks": int(best_for_sym["lag"]),
                "best_lag_seconds": int(best_for_sym["lag"]) * 2,
                "corr": round(best_for_sym["corr"], 3),
                "recent_move": round(recent_move, 2),
            }

            if abs(best_for_sym["corr"]) > best_abs_corr:
                best_abs_corr = abs(best_for_sym["corr"])
                best_corr = best_for_sym["corr"]
                best_lag = int(best_for_sym["lag"])
                best_sym = sym

        if best_sym is not None:
            result["best_leader"] = {
                "symbol": best_sym,
                "lag_ticks": best_lag,
                "lag_seconds": best_lag * 2,
                "corr": round(best_corr, 3),
            }

        # ── Emit directional signals for strong leaders with real moves ──
        if best_sym is not None and best_abs_corr >= CORR_CONFIDENCE:
            leader_data = result["leaders"][best_sym]
            move = leader_data["recent_move"]
            if abs(move) >= MIN_LEADER_MOVE:
                # Sign of WDO's expected follow = sign(corr) × sign(leader_move)
                expected_sign = (1 if best_corr > 0 else -1) * (1 if move > 0 else -1)
                if best_sym == "IND":
                    if expected_sign > 0:
                        reasons.append("leadlag_ind_bull")
                    else:
                        reasons.append("leadlag_ind_bear")
                    adj += 0.07
                elif best_sym == "ES":
                    if expected_sign > 0:
                        reasons.append("leadlag_es_bull")
                    else:
                        reasons.append("leadlag_es_bear")
                    adj += 0.06
                elif best_sym == "NQ":
                    reasons.append("leadlag_nq_signal")
                    adj += 0.04
                result["signal"] = f"{best_sym.lower()}_leading"

        # ── No leader — all correlations weak ──
        if best_abs_corr < CORR_CONFIDENCE and len(result["leaders"]) >= 2:
            reasons.append("leadlag_no_leader")
            # Slight positive adj — idiosyncratic regime favors local signals
            adj += 0.02
            result["signal"] = "idiosyncratic"

        result["score_adjustment"] = round(adj, 3)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Dump state for API/frontend."""
        return {
            "date": self._state.date,
            "wdo_samples": len(self._state.wdo_prices),
            "leader_samples": {
                sym: len(ld.prices) for sym, ld in self._state.leaders.items()
            },
        }


# Singleton accessor
_leadlag_instance: Optional[LeadLagDetector] = None


def get_leadlag_detector() -> LeadLagDetector:
    """Get the singleton lead-lag detector."""
    global _leadlag_instance
    if _leadlag_instance is None:
        _leadlag_instance = LeadLagDetector()
    return _leadlag_instance
