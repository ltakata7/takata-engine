"""Implied Volatility module — computes IV, Greeks, and vol signals from market data.

Uses py_vollib for Black-Scholes pricing and IV extraction.
Integrates with VXBR (Brazilian VIX) from the Profit Ultra bridge.

Key signals:
- High IV = fear/uncertainty → wider stops, smaller size or no trade
- IV crush = post-event calm → trending environment, follow momentum
- IV skew = directional fear → put skew = bearish sentiment
- IV term structure → contango = calm, backwardation = stress
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── VXBR History Tracker (singleton state) ──────────────────────────
# Stores rolling VXBR readings for percentile calculation and rate-of-change.
# ~7 hours of 2-second ticks = 12,600 readings per day.

_VXBR_HISTORY_MAX = 15000  # ~8+ hours at 2s intervals
_vxbr_history: deque = deque(maxlen=_VXBR_HISTORY_MAX)
_vxbr_last_ts: float = 0.0
_vxbr_daily_reset: str = ""


@dataclass
class VolRegime:
    """Volatility regime classification from IV levels."""
    vxbr: float           # Brazilian VIX level
    iv_percentile: float  # IV percentile vs history (0-100)
    regime: str           # "low", "normal", "elevated", "extreme"
    signal: str           # "risk_on", "cautious", "defensive", "no_trade"
    size_multiplier: float  # 1.0 = full, 0.5 = half, 0.0 = no trade
    stop_multiplier: float  # 1.0 = normal, 1.5 = wider, 2.0 = very wide


def classify_vol_regime(vxbr: float) -> VolRegime:
    """Classify the volatility regime from VXBR level.

    VXBR thresholds calibrated for Brazilian market:
    - < 15: Low vol → risk on, trend following
    - 15-22: Normal → standard trading
    - 22-30: Elevated → reduce size, widen stops
    - > 30: Extreme → defensive or no trade

    Parameters
    ----------
    vxbr : float
        Current VXBR (Brazilian VIX) level.

    Returns
    -------
    VolRegime
    """
    if vxbr <= 0:
        return VolRegime(vxbr, 50, "unknown", "cautious", 1.0, 1.0)

    if vxbr < 15:
        return VolRegime(vxbr, 20, "low", "risk_on", 1.0, 1.0)
    elif vxbr < 22:
        return VolRegime(vxbr, 50, "normal", "cautious", 1.0, 1.0)
    elif vxbr < 30:
        return VolRegime(vxbr, 75, "elevated", "defensive", 0.5, 1.5)
    else:
        return VolRegime(vxbr, 95, "extreme", "no_trade", 0.0, 2.0)


class VXBRTracker:
    """Tracks VXBR intraday for percentile-based sizing and rate-of-change signals.

    Call ``update(vxbr)`` every scan cycle. Call ``signal()`` for the full analysis.
    Resets history daily. Persists intraday readings in a module-level deque.
    """

    def update(self, vxbr: float) -> None:
        """Record a new VXBR reading."""
        global _vxbr_last_ts, _vxbr_daily_reset
        if vxbr <= 0:
            return

        now = time.time()
        today = time.strftime("%Y-%m-%d")

        # Day reset
        if _vxbr_daily_reset != today:
            _vxbr_history.clear()
            _vxbr_daily_reset = today

        _vxbr_history.append((now, vxbr))
        _vxbr_last_ts = now

    def signal(self, vxbr: float) -> Dict[str, Any]:
        """Full VXBR analysis: regime, continuous sizing, percentile, rate of change.

        Returns dict with all fields from vol_signal_for_scanner() plus:
        - ``percentile_intraday``: where current VXBR sits vs today's range (0-100)
        - ``rate_of_change_5m``: VXBR change over last 5 min
        - ``rate_of_change_30m``: VXBR change over last 30 min
        - ``trend``: rising / falling / stable
        - ``size_continuous``: smooth 0.0-1.0 sizing (not discrete tiers)
        - ``stop_continuous``: smooth 1.0-2.5 stop multiplier
        - ``target_continuous``: smooth 1.0-2.0 target multiplier
        - ``score_adjustment``: signal score adjustment based on vol dynamics
        - ``reasons``: list of vol-based reasons
        """
        base = classify_vol_regime(vxbr)
        result: Dict[str, Any] = {
            "vxbr": vxbr,
            "vol_regime": base.regime,
            "vol_signal": base.signal,
            "size_multiplier": base.size_multiplier,
            "stop_multiplier": base.stop_multiplier,
            "score_adjustment": 0.0,
            "reasons": [],
        }

        if vxbr <= 0:
            return result

        reasons = result["reasons"]
        adj = 0.0

        # ── Continuous sizing (sigmoid curves, no discrete jumps) ──
        # Size: 1.0 at VXBR=10, 0.5 at VXBR=25, ~0 at VXBR=32
        size_cont = 1.0 / (1.0 + np.exp(0.3 * (vxbr - 25)))
        size_cont = max(0.0, min(1.0, float(size_cont)))

        # Stops: 1.0 at low vol, 2.5 at extreme vol
        stop_cont = 1.0 + 1.5 / (1.0 + np.exp(-0.2 * (vxbr - 22)))
        stop_cont = max(1.0, min(2.5, float(stop_cont)))

        # Targets: higher vol = wider targets (capture bigger moves)
        target_cont = 1.0 + 1.0 / (1.0 + np.exp(-0.2 * (vxbr - 20)))
        target_cont = max(1.0, min(2.0, float(target_cont)))

        result["size_continuous"] = round(size_cont, 3)
        result["stop_continuous"] = round(stop_cont, 3)
        result["target_continuous"] = round(target_cont, 3)

        # Override discrete multipliers with continuous values
        result["size_multiplier"] = round(size_cont, 3)
        result["stop_multiplier"] = round(stop_cont, 3)

        # ── Intraday percentile ──
        if len(_vxbr_history) >= 10:
            values = [v for _, v in _vxbr_history]
            pct = float(np.searchsorted(np.sort(values), vxbr) / len(values) * 100)
            result["percentile_intraday"] = round(pct, 1)

            # Very high percentile = VXBR spiking today
            if pct >= 90:
                reasons.append("vxbr_intraday_spike")
                adj -= 0.06
            elif pct <= 10:
                reasons.append("vxbr_intraday_low")
                adj += 0.04
        else:
            result["percentile_intraday"] = 50.0

        # ── Rate of change ──
        roc_5m = self._rate_of_change(300)   # 5 min
        roc_30m = self._rate_of_change(1800)  # 30 min
        result["rate_of_change_5m"] = round(roc_5m, 2) if roc_5m is not None else None
        result["rate_of_change_30m"] = round(roc_30m, 2) if roc_30m is not None else None

        # ── Trend classification ──
        trend = "stable"
        if roc_5m is not None and roc_30m is not None:
            if roc_5m > 1.0 and roc_30m > 1.5:
                trend = "rising_fast"
                reasons.append("vxbr_rising_fast")
                adj -= 0.08  # vol spiking = reduce signal confidence
            elif roc_5m > 0.5 or roc_30m > 1.0:
                trend = "rising"
                reasons.append("vxbr_rising")
                adj -= 0.04
            elif roc_5m < -1.0 and roc_30m < -1.5:
                trend = "falling_fast"
                reasons.append("vxbr_falling_fast")
                adj += 0.06  # vol crushing = better signal environment
            elif roc_5m < -0.5 or roc_30m < -1.0:
                trend = "falling"
                reasons.append("vxbr_falling")
                adj += 0.03
        elif roc_5m is not None:
            if roc_5m > 0.5:
                trend = "rising"
            elif roc_5m < -0.5:
                trend = "falling"

        result["trend"] = trend

        # ── Vol crush detection (IV dropping from elevated) ──
        if trend in ("falling", "falling_fast") and vxbr > 18 and roc_30m is not None and roc_30m < -2.0:
            reasons.append("vxbr_crush")
            adj += 0.08  # post-event crush = strong trending environment

        result["score_adjustment"] = round(adj, 3)
        return result

    def _rate_of_change(self, lookback_seconds: int) -> Optional[float]:
        """VXBR change over the last N seconds."""
        if len(_vxbr_history) < 2:
            return None

        now = time.time()
        cutoff = now - lookback_seconds

        # Find the reading closest to the cutoff
        old_val = None
        for ts, val in _vxbr_history:
            if ts >= cutoff:
                old_val = val
                break

        if old_val is None:
            return None

        current = _vxbr_history[-1][1]
        return current - old_val


# Module-level singleton
_vxbr_tracker: Optional[VXBRTracker] = None


def get_vxbr_tracker() -> VXBRTracker:
    """Get or create the VXBR tracker singleton."""
    global _vxbr_tracker
    if _vxbr_tracker is None:
        _vxbr_tracker = VXBRTracker()
    return _vxbr_tracker


def compute_greeks(
    spot: float,
    strike: float,
    days_to_expiry: int,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "c",
) -> Dict[str, float]:
    """Compute Black-Scholes Greeks for a single option.

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike : float
        Option strike price.
    days_to_expiry : int
        Days until expiration.
    risk_free_rate : float
        Risk-free rate (annualized, e.g. 0.136 for 13.6%).
    volatility : float
        Implied volatility (annualized, e.g. 0.12 for 12%).
    option_type : str
        ``"c"`` for call, ``"p"`` for put.

    Returns
    -------
    dict
        price, delta, gamma, theta, vega, rho
    """
    from py_vollib.black_scholes import black_scholes as bs
    from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega, rho

    t = max(days_to_expiry / 365.0, 0.001)

    try:
        return {
            "price": bs(option_type, spot, strike, t, risk_free_rate, volatility),
            "delta": delta(option_type, spot, strike, t, risk_free_rate, volatility),
            "gamma": gamma(option_type, spot, strike, t, risk_free_rate, volatility),
            "theta": theta(option_type, spot, strike, t, risk_free_rate, volatility),
            "vega": vega(option_type, spot, strike, t, risk_free_rate, volatility),
            "rho": rho(option_type, spot, strike, t, risk_free_rate, volatility),
        }
    except Exception as e:
        logger.warning("Greeks computation failed: %s", e)
        return {"price": 0, "delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}


def compute_implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    days_to_expiry: int,
    risk_free_rate: float,
    option_type: str = "c",
) -> float:
    """Extract implied volatility from an option price.

    Parameters
    ----------
    option_price : float
        Market price of the option.
    spot, strike, days_to_expiry, risk_free_rate, option_type :
        Standard BS parameters.

    Returns
    -------
    float
        Implied volatility (annualized).
    """
    from py_vollib.black_scholes.implied_volatility import implied_volatility

    t = max(days_to_expiry / 365.0, 0.001)

    try:
        return implied_volatility(option_price, spot, strike, t, risk_free_rate, option_type)
    except Exception as e:
        logger.warning("IV computation failed: %s", e)
        return 0.0


def covered_call_analysis(
    spot: float,
    strike: float,
    days_to_expiry: int,
    risk_free_rate: float,
    volatility: float,
    shares: int = 100,
) -> Dict[str, Any]:
    """Analyze a covered call position for Sympatheia Advisory clients.

    Parameters
    ----------
    spot : float
        Current stock/ETF price.
    strike : float
        Call strike to sell.
    days_to_expiry : int
        Days until expiration.
    risk_free_rate : float
        Risk-free rate.
    volatility : float
        Implied volatility.
    shares : int
        Number of shares held.

    Returns
    -------
    dict
        Premium, yield, max profit, breakeven, protection
    """
    greeks = compute_greeks(spot, strike, days_to_expiry, risk_free_rate, volatility, "c")
    premium = greeks["price"]

    premium_per_share = premium
    total_premium = premium * shares
    annualized_yield = (premium_per_share / spot) * (365 / max(days_to_expiry, 1)) * 100
    max_profit = (strike - spot + premium_per_share) * shares
    breakeven = spot - premium_per_share
    downside_protection = premium_per_share / spot * 100

    return {
        "premium_per_share": round(premium_per_share, 2),
        "total_premium": round(total_premium, 2),
        "annualized_yield": round(annualized_yield, 1),
        "max_profit": round(max_profit, 2),
        "breakeven": round(breakeven, 2),
        "downside_protection": round(downside_protection, 1),
        "delta": round(greeks["delta"], 3),
        "days_to_expiry": days_to_expiry,
    }


def vol_signal_for_scanner(vxbr: float) -> Dict[str, Any]:
    """Generate a volatility signal for the live scanner.

    Parameters
    ----------
    vxbr : float
        Current VXBR level.

    Returns
    -------
    dict
        vol_regime, size_multiplier, stop_multiplier, signal
    """
    regime = classify_vol_regime(vxbr)
    return {
        "vxbr": regime.vxbr,
        "vol_regime": regime.regime,
        "vol_signal": regime.signal,
        "size_multiplier": regime.size_multiplier,
        "stop_multiplier": regime.stop_multiplier,
    }
