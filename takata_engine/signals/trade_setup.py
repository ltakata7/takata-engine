"""Trade Setup — actionable trade ideas with fixed levels and lifecycle.

Unlike continuous signals that recalculate every scan, a TradeSetup:
- Triggers ONCE when confluence is high enough
- Locks in fixed entry zone, stop loss, and profit targets
- Stays active until taken, invalidated, or expired
- Tracks partial profit levels (T1 = 50%, T2 = remaining)

Lifecycle: pending → active → taken/invalidated/expired
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Setup validity (seconds)
DEFAULT_EXPIRY = 300  # 5 minutes


@dataclass
class TradeSetup:
    """An actionable trade setup with fixed levels."""

    # Identity
    setup_id: str = ""
    instrument: str = ""
    direction: str = ""  # "long" or "short"
    created_at: str = ""

    # Fixed levels (set once, never change)
    entry_low: float = 0      # bottom of entry zone
    entry_high: float = 0     # top of entry zone
    stop_loss: float = 0      # hard stop — invalidation level
    target_1: float = 0       # partial profit (50%) — 1:1 R:R or next level
    target_2: float = 0       # full profit (remaining 50%) — 2:1 R:R or second level
    breakeven_after_t1: bool = True  # move stop to entry after T1 hit

    # Risk metrics (computed at creation)
    risk_pts: float = 0       # distance to stop in points
    reward_1_pts: float = 0   # distance to T1
    reward_2_pts: float = 0   # distance to T2
    rr_ratio: float = 0       # reward-to-risk (T2/risk)
    risk_per_contract: float = 0  # R$ or $ risk per contract
    suggested_contracts: int = 0

    # Context at creation
    score: float = 0          # signal score when setup triggered
    reasons: List[str] = field(default_factory=list)
    regime: str = ""
    session_phase: str = ""
    bias: str = ""
    indicators: Dict[str, float] = field(default_factory=dict)

    # Level sources (for transparency)
    stop_source: str = ""     # e.g., "daily_support + ATR buffer"
    t1_source: str = ""       # e.g., "1:1 R:R"
    t2_source: str = ""       # e.g., "pivot_R1"

    # Lifecycle
    status: str = "active"    # "active", "taken", "invalidated", "expired"
    expires_at: float = 0     # unix timestamp
    invalidated_reason: str = ""
    taken_at: str = ""
    signal_id: str = ""       # link to ML signal logger

    def __post_init__(self):
        if not self.setup_id:
            self.setup_id = f"{self.instrument}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.expires_at:
            self.expires_at = time.time() + DEFAULT_EXPIRY

    @property
    def is_active(self) -> bool:
        return self.status == "active" and time.time() < self.expires_at

    @property
    def seconds_remaining(self) -> float:
        return max(0, self.expires_at - time.time())

    def check_invalidation(self, current_price: float) -> bool:
        """Check if price has broken through the stop level before entry.
        If so, the setup is no longer valid.
        """
        if self.status != "active":
            return False

        # Expired?
        if time.time() >= self.expires_at:
            self.status = "expired"
            self.invalidated_reason = "timeout"
            return True

        # Price broke through stop before entry was taken
        if self.direction == "long" and current_price <= self.stop_loss:
            self.status = "invalidated"
            self.invalidated_reason = "price_hit_stop_before_entry"
            return True
        elif self.direction == "short" and current_price >= self.stop_loss:
            self.status = "invalidated"
            self.invalidated_reason = "price_hit_stop_before_entry"
            return True

        return False

    def mark_taken(self) -> None:
        self.status = "taken"
        self.taken_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "instrument": self.instrument,
            "direction": self.direction,
            "created_at": self.created_at,
            "entry_low": self.entry_low,
            "entry_high": self.entry_high,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "target_2": self.target_2,
            "breakeven_after_t1": self.breakeven_after_t1,
            "risk_pts": self.risk_pts,
            "reward_1_pts": self.reward_1_pts,
            "reward_2_pts": self.reward_2_pts,
            "rr_ratio": round(self.rr_ratio, 2),
            "risk_per_contract": self.risk_per_contract,
            "suggested_contracts": self.suggested_contracts,
            "score": self.score,
            "reasons": self.reasons,
            "regime": self.regime,
            "session_phase": self.session_phase,
            "bias": self.bias,
            "stop_source": self.stop_source,
            "t1_source": self.t1_source,
            "t2_source": self.t2_source,
            "status": self.status if self.is_active or self.status != "active" else "expired",
            "seconds_remaining": round(self.seconds_remaining, 0),
            "invalidated_reason": self.invalidated_reason,
            "taken_at": self.taken_at,
            "signal_id": self.signal_id,
        }


def build_wdo_setup(
    price: float,
    direction: str,
    score: float,
    reasons: List[str],
    indicators: Dict[str, float],
    session_phase: str,
    bias: str,
    regime: str,
    levels: Dict[str, Any],
    atr: float,
    vol_stop_mult: float = 1.0,
    max_risk_brl: float = 2000.0,
    **kwargs,
) -> Optional[TradeSetup]:
    """Build a WDO trade setup calibrated for scalping with reasonable stops.

    WDO scalping philosophy (bigger lot, smaller points, REASONABLE stops):
    - Stop: 3-5 pts (above WDO noise floor of 1-3pts, R$30-50/ct)
    - T1: ~half the risk (1.5-2.5 pts, quick partial → move stop to BE)
    - T2: 1.5-2x risk (4-7 pts full target)
    - Minimum R:R of 1.5:1 (scalping tolerates tighter R:R with higher hit rate)
    - Lot sizing: scales up with tight stops (max R$2,000 risk → up to 80 contracts)
    - Trade entire session: no phase blocks

    Example math:
      3pt stop × R$10 × 60 contracts = R$1,800 risk
      5pt target × R$10 × 60 contracts = R$3,000 profit (1.67:1)
    """
    tick = 0.5   # WDO tick size
    point_value = 10.0  # R$10 per point per contract

    # ── WDO scalp constraints — reasonable stops above noise floor ──
    MIN_STOP = 3.0    # pts — R$30/ct (above 1-3pt noise)
    MAX_STOP = 5.0    # pts — R$50/ct (max for scalping)
    MIN_RR = 1.5      # min 1.5:1
    MIN_T2 = 4.0      # minimum full target
    MAX_T2 = 7.0      # cap full target (high hit rate priority)
    MAX_CONTRACTS = 100  # max lot size

    # ── Use real intraday vol to pick stop within 3-5 range ──
    iv = kwargs.get("intraday_vol", {})
    if iv and iv.get("bar_count", 0) >= 5:
        # Scale stop by recent candle range — needs to survive normal noise
        recent = iv["recent_range"]
        if recent < 5:
            stop_base = 3.0
        elif recent < 8:
            stop_base = 4.0
        else:
            stop_base = 5.0
        market_type = iv.get("market_type", "normal")
        vol_info = f"real({recent:.1f}avg, {market_type})"
    else:
        stop_base = max(MIN_STOP, min(atr * 0.7, MAX_STOP))
        market_type = "unknown"
        vol_info = f"ATR({atr:.1f})"

    # Target multiplier: 1.5:1 for scalping (smaller targets = higher hit rate)
    target_mult_from_vol = 1.5

    # ── Implied vol adjustment — still within scalp limits ──
    vxbr = kwargs.get("vxbr", 0)
    hist_vol = indicators.get("hist_vol", 0) if indicators else 0
    if vxbr > 0 and hist_vol > 0:
        iv_rv_ratio = vxbr / hist_vol
        if iv_rv_ratio > 1.3:
            stop_base = min(stop_base + 0.5, MAX_STOP)
            vol_info += f" IV/RV={iv_rv_ratio:.1f}↑"
        elif iv_rv_ratio < 0.7:
            stop_base = max(stop_base - 0.5, MIN_STOP)
            vol_info += f" IV/RV={iv_rv_ratio:.1f}↓"
    elif vxbr > 25:
        stop_base = min(stop_base + 0.5, MAX_STOP)
        vol_info += f" VXBR={vxbr:.0f}↑"

    # ── Entry zone: tight, based on recent candle range ──
    recent_range = iv.get("recent_range", atr) if iv else atr
    entry_buffer = round(max(recent_range * 0.2, 0.5) / tick) * tick
    entry_low = round((price - entry_buffer) / tick) * tick
    entry_high = round((price + entry_buffer) / tick) * tick

    # ── Stop loss: from intraday vol, capped ──
    stop_dist = max(stop_base, MIN_STOP)
    stop_dist = min(stop_dist, MAX_STOP)
    stop_dist = round(stop_dist * 2) / 2  # tick-align
    stop_source = f"{vol_info}→{stop_dist}pts"

    # Tighten stop if a nearby institutional level provides protection
    if direction == "long":
        s1 = levels.get("s1", 0)
        session_low = levels.get("session_low", 0)
        # If S1 or session low is within our ATR-based stop, use it (tighter = better)
        for lvl, name in [(s1, "S1"), (session_low, "session_low")]:
            if lvl and 0 < price - lvl < stop_dist and price - lvl >= MIN_STOP:
                level_stop = round((price - lvl + 1.0) / tick) * tick  # 1pt buffer below level
                if level_stop < stop_dist:
                    stop_dist = level_stop
                    stop_source = f"{name}({lvl})+1pt"
        stop = round((price - stop_dist) / tick) * tick
    else:
        r1 = levels.get("r1", 0)
        session_high = levels.get("session_high", 0)
        for lvl, name in [(r1, "R1"), (session_high, "session_high")]:
            if lvl and 0 < lvl - price < stop_dist and lvl - price >= MIN_STOP:
                level_stop = round((lvl - price + 1.0) / tick) * tick
                if level_stop < stop_dist:
                    stop_dist = level_stop
                    stop_source = f"{name}({lvl})+1pt"
        stop = round((price + stop_dist) / tick) * tick

    risk_pts = stop_dist

    # ── Target 1: ~half the risk (quick partial profit + move stop to BE) ──
    # For 3pt stop → T1 at 1.5pts; for 5pt stop → T1 at 2.5pts
    t1_dist = max(1.5, round(risk_pts * 0.5 * 2) / 2)
    rr1_price = price + t1_dist if direction == "long" else price - t1_dist
    t1 = round(rr1_price / tick) * tick
    t1_source = f"0.5R ({t1_dist}pts)"

    # ── Target 2: 1.5:1 R:R for scalping, capped within MIN_T2..MAX_T2 ──
    target_dist = max(MIN_T2, min(risk_pts * target_mult_from_vol, MAX_T2))
    target_dist = round(target_dist * 2) / 2  # tick-align
    rr2_price = price + target_dist if direction == "long" else price - target_dist
    t2 = round(rr2_price / tick) * tick
    t2_source = f"{target_mult_from_vol}:1_RR ({target_dist}pts)"

    reward_1 = abs(t1 - price)
    reward_2 = abs(t2 - price)
    rr_ratio = reward_2 / risk_pts if risk_pts > 0 else 0

    # ── Reject if R:R too low ──
    if rr_ratio < MIN_RR:
        logger.debug("WDO setup rejected: R:R %.2f < %.1f minimum", rr_ratio, MIN_RR)
        return None

    # ── Contract sizing — scale up for tight stops ──
    risk_per_ct = risk_pts * point_value
    suggested_cts = int(max_risk_brl / risk_per_ct) if risk_per_ct > 0 else 50
    suggested_cts = max(10, min(suggested_cts, MAX_CONTRACTS))

    return TradeSetup(
        instrument="WDO",
        direction=direction,
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=stop,
        target_1=t1,
        target_2=t2,
        risk_pts=round(risk_pts, 1),
        reward_1_pts=round(reward_1, 1),
        reward_2_pts=round(reward_2, 1),
        rr_ratio=rr_ratio,
        risk_per_contract=round(risk_per_ct, 0),
        suggested_contracts=suggested_cts,
        score=score,
        reasons=reasons,
        regime=regime,
        session_phase=session_phase,
        bias=bias,
        indicators=indicators,
        stop_source=stop_source,
        t1_source=t1_source,
        t2_source=t2_source,
    )


def build_mes_setup(
    price: float,
    direction: str,
    score: float,
    reasons: List[str],
    indicators: Dict[str, float],
    session_phase: str,
    bias: str,
    regime: str,
    levels: Dict[str, Any],
    atr: float,
    max_risk_usd: float = 200.0,
) -> Optional[TradeSetup]:
    """Build an MES trade setup with fixed levels."""
    tick = 0.25
    point_value = 5.0

    entry_buffer = round(atr * 0.2 / tick) * tick
    entry_low = round((price - entry_buffer) / tick) * tick
    entry_high = round((price + entry_buffer) / tick) * tick

    # Stop from institutional levels
    atr_buffer = round(atr * 0.5 / tick) * tick
    stop = 0.0
    stop_source = ""

    daily_sup = levels.get("daily_support", [])
    daily_res = levels.get("daily_resistance", [])
    prev_low = levels.get("prev_day_low", 0)
    prev_high = levels.get("prev_day_high", 0)

    if direction == "long":
        supports = [s for s in (daily_sup or []) + [prev_low] if s and 0 < s < price]
        if supports:
            nearest = max(supports)
            stop = round((nearest - atr_buffer) / tick) * tick
            stop_source = f"support({nearest}) - buffer"
        else:
            stop_dist = max(atr * 1.5, 3.0)
            stop_dist = min(stop_dist, 20.0)
            stop = round((price - stop_dist) / tick) * tick
            stop_source = f"ATR-based({stop_dist}pts)"
    else:
        resistances = [r for r in (daily_res or []) + [prev_high] if r and r > price]
        if resistances:
            nearest = min(resistances)
            stop = round((nearest + atr_buffer) / tick) * tick
            stop_source = f"resistance({nearest}) + buffer"
        else:
            stop_dist = max(atr * 1.5, 3.0)
            stop_dist = min(stop_dist, 20.0)
            stop = round((price + stop_dist) / tick) * tick
            stop_source = f"ATR-based({stop_dist}pts)"

    risk_pts = abs(price - stop)
    if risk_pts < 2.0:
        risk_pts = 3.0
        stop = round((price - 3.0 if direction == "long" else price + 3.0) / tick) * tick
        stop_source = "min_stop(3pts)"
    if risk_pts > 25.0:
        return None

    # Targets
    rr1 = price + risk_pts if direction == "long" else price - risk_pts
    rr2 = price + risk_pts * 2 if direction == "long" else price - risk_pts * 2
    t1 = round(rr1 / tick) * tick
    t2 = round(rr2 / tick) * tick
    t1_source = "1:1_RR"
    t2_source = "2:1_RR"

    # Use institutional levels if available
    if direction == "long" and daily_res:
        above = [r for r in daily_res if r > price]
        if above:
            if above[0] < rr1 * 1.1:
                t1 = round(above[0] / tick) * tick
                t1_source = f"resistance({above[0]})"
            if len(above) > 1 and above[1] < rr2 * 1.3:
                t2 = round(above[1] / tick) * tick
                t2_source = f"resistance({above[1]})"

    reward_1 = abs(t1 - price)
    reward_2 = abs(t2 - price)
    rr_ratio = reward_2 / risk_pts if risk_pts > 0 else 0

    risk_per_ct = risk_pts * point_value
    suggested_cts = int(max_risk_usd / risk_per_ct) if risk_per_ct > 0 else 1
    suggested_cts = max(1, min(suggested_cts, 2))

    return TradeSetup(
        instrument="MES",
        direction=direction,
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=stop,
        target_1=t1,
        target_2=t2,
        risk_pts=round(risk_pts, 2),
        reward_1_pts=round(reward_1, 2),
        reward_2_pts=round(reward_2, 2),
        rr_ratio=rr_ratio,
        risk_per_contract=round(risk_per_ct, 0),
        suggested_contracts=suggested_cts,
        score=score,
        reasons=reasons,
        regime=regime,
        session_phase=session_phase,
        bias=bias,
        indicators=indicators,
        stop_source=stop_source,
        t1_source=t1_source,
        t2_source=t2_source,
    )
