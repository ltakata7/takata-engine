"""Realized-vs-Implied Volatility ratio tracker for WDO fast-timeframe trading.

Computes rolling tick-based realized volatility and compares it to the VXBR
implied-volatility reading. The RV/IV ratio is one of the most reliable
mean-reversion signals institutional desks use intraday.

**Why RV vs IV matters for WDO 3-5min trading:**
- RV/IV << 1: market is quiet but options are pricing fear → fade vol,
  favor mean reversion, expect breakouts to fail
- RV/IV >> 1: actual moves exceed what options imply → genuine vol shock
  → momentum trades dominate, let trends run, widen stops
- RV/IV ≈ 1: efficient market, no edge from vol dispersion
- The ratio itself mean-reverts on a multi-hour timescale → extremes fade

Signal types generated:
- ``rv_below_iv_strong`` — RV/IV < 0.6 = volatility premium, fade momentum
- ``rv_below_iv``        — RV/IV < 0.8 = mild premium, mean reversion setup
- ``rv_above_iv``        — RV/IV > 1.2 = realized exceeding implied, ride trends
- ``rv_above_iv_shock``  — RV/IV > 1.8 = vol shock regime, size down, chase momentum
- ``rv_rising``          — realized vol trending up on a 5-min window
- ``rv_falling``         — realized vol collapsing = compression before move

The tracker maintains a 5-min tick price window and also the 15-min RV for
context. Annualization assumes 250 trading days × 6.5 hours × 60 min for
intraday-scaled output.

Call ``update()`` every scan cycle with the current price.
Call ``signal(vxbr)`` with the current VXBR reading to get the signal dict.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import pytz

logger = logging.getLogger(__name__)

BRT = pytz.timezone("America/Sao_Paulo")

# Tick windows for realized-vol computation.
# Scanner ticks roughly every 2s → 5 min ≈ 150 samples, 15 min ≈ 450 samples.
RV_WINDOW_5M = 150
RV_WINDOW_15M = 450

# Annualization: ~250 trading days × 6.5 hours × 60 min = ~97,500 min/yr
# For tick-based scaling we normalize to "per-minute" and then to annual.
MINUTES_PER_YEAR = 250 * 6.5 * 60
SECONDS_PER_MINUTE = 60

# Threshold for a meaningful ratio reading (need enough ticks)
MIN_TICKS_FOR_RV = 30

# History of (timestamp, rv_5m, vxbr, ratio) for trend detection
RATIO_HISTORY_LEN = 120


@dataclass
class RVState:
    """Realized-vol tracker state — rolling price buffers."""
    price_window_5m: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=RV_WINDOW_5M)
    )  # (ts, price)
    price_window_15m: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=RV_WINDOW_15M)
    )
    # History of ratio readings for trend detection
    ratio_history: Deque[Tuple[float, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=RATIO_HISTORY_LEN)
    )  # (ts, rv_5m, vxbr, ratio)


def _annualized_rv(prices: List[Tuple[float, float]]) -> float:
    """Annualized realized volatility from a list of (ts, price) samples.

    Uses log-returns, computes sample std, then scales to annualized %.
    Returns 0 if not enough samples or all prices equal.
    """
    if len(prices) < MIN_TICKS_FOR_RV:
        return 0.0
    # Build return series
    rets: List[float] = []
    for i in range(1, len(prices)):
        p_prev = prices[i - 1][1]
        p_cur = prices[i][1]
        if p_prev > 0 and p_cur > 0:
            rets.append(math.log(p_cur / p_prev))
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    std = math.sqrt(max(var, 0.0))
    # Infer sampling interval from timestamps
    dt_total = prices[-1][0] - prices[0][0]
    if dt_total <= 0:
        return 0.0
    seconds_per_sample = dt_total / (len(prices) - 1)
    if seconds_per_sample <= 0:
        return 0.0
    samples_per_year = (MINUTES_PER_YEAR * SECONDS_PER_MINUTE) / seconds_per_sample
    annualized = std * math.sqrt(samples_per_year) * 100.0
    return annualized


class RVIVTracker:
    """Realized-vs-implied vol ratio tracker.

    Complements the VXBR tracker: VXBR tells you the fear premium level,
    this tracker tells you whether the market is actually delivering on
    that premium.
    """

    def __init__(self) -> None:
        self._state = RVState()

    def update(self, price: float) -> None:
        """Record a new tick price.

        Parameters
        ----------
        price : float
            Current trade price.
        """
        if price <= 0:
            return
        now = datetime.now(BRT).timestamp()
        self._state.price_window_5m.append((now, price))
        self._state.price_window_15m.append((now, price))

    def signal(self, vxbr: float) -> Dict[str, Any]:
        """Generate RV-vs-IV signal dict.

        Parameters
        ----------
        vxbr : float
            Current VXBR (implied vol) reading.
        """
        result: Dict[str, Any] = {
            "rv_5m": 0.0,
            "rv_15m": 0.0,
            "iv": round(vxbr, 2) if vxbr else 0,
            "ratio_5m": 0.0,
            "ratio_15m": 0.0,
            "regime": "insufficient_data",
            "signal": "neutral",
            "score_adjustment": 0.0,
            "reasons": [],
        }
        reasons: List[str] = result["reasons"]
        adj = 0.0

        if vxbr <= 0:
            return result

        rv_5m = _annualized_rv(list(self._state.price_window_5m))
        rv_15m = _annualized_rv(list(self._state.price_window_15m))
        result["rv_5m"] = round(rv_5m, 2)
        result["rv_15m"] = round(rv_15m, 2)

        if rv_5m <= 0:
            return result

        ratio_5m = rv_5m / vxbr
        ratio_15m = rv_15m / vxbr if rv_15m > 0 else 0
        result["ratio_5m"] = round(ratio_5m, 3)
        result["ratio_15m"] = round(ratio_15m, 3)

        # Record for trend detection
        now = datetime.now(BRT).timestamp()
        self._state.ratio_history.append((now, rv_5m, vxbr, ratio_5m))

        # ── Regime classification ──
        if ratio_5m < 0.6:
            result["regime"] = "vol_premium_strong"
            reasons.append("rv_below_iv_strong")
            adj += 0.10  # mean reversion favored, fade breakouts
        elif ratio_5m < 0.8:
            result["regime"] = "vol_premium"
            reasons.append("rv_below_iv")
            adj += 0.05
        elif ratio_5m > 1.8:
            result["regime"] = "vol_shock"
            reasons.append("rv_above_iv_shock")
            adj += 0.08  # momentum regime, ride moves
            result["signal"] = "momentum_regime"
        elif ratio_5m > 1.2:
            result["regime"] = "vol_elevated"
            reasons.append("rv_above_iv")
            adj += 0.05
            result["signal"] = "momentum_regime"
        else:
            result["regime"] = "efficient"

        # ── RV trend (5m RV change over last 60 samples ≈ 2 min) ──
        history = list(self._state.ratio_history)
        if len(history) >= 30:
            lookback = max(0, len(history) - 30)
            _, ref_rv, _, _ = history[lookback]
            if ref_rv > 0:
                rv_change_pct = (rv_5m - ref_rv) / ref_rv
                result["rv_change_pct"] = round(rv_change_pct, 3)
                if rv_change_pct > 0.25:
                    reasons.append("rv_rising")
                    adj += 0.05  # vol expansion = moves incoming
                elif rv_change_pct < -0.25:
                    reasons.append("rv_falling")
                    adj += 0.04  # compression often precedes breakouts

        result["score_adjustment"] = round(adj, 3)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Dump current state for API/frontend."""
        latest_5m = 0.0
        latest_15m = 0.0
        if len(self._state.price_window_5m) >= MIN_TICKS_FOR_RV:
            latest_5m = round(_annualized_rv(list(self._state.price_window_5m)), 2)
        if len(self._state.price_window_15m) >= MIN_TICKS_FOR_RV:
            latest_15m = round(_annualized_rv(list(self._state.price_window_15m)), 2)
        return {
            "rv_5m": latest_5m,
            "rv_15m": latest_15m,
            "window_5m_depth": len(self._state.price_window_5m),
            "window_15m_depth": len(self._state.price_window_15m),
            "history_depth": len(self._state.ratio_history),
        }


# Singleton accessor
_rv_iv_tracker_instance: Optional[RVIVTracker] = None


def get_rv_iv_tracker() -> RVIVTracker:
    """Get the singleton RV/IV tracker."""
    global _rv_iv_tracker_instance
    if _rv_iv_tracker_instance is None:
        _rv_iv_tracker_instance = RVIVTracker()
    return _rv_iv_tracker_instance
