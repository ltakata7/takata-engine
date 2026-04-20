"""Live DI Curve — builds multi-tenor curve from Profit Ultra bridge data.

Fetches all DI futures contracts from the bridge, maps them to business days,
and builds a full curve using BNDF5 exponential interpolation.

Curve shape analysis provides macro signals for WDO:
- Steepening: market expects higher rates → BRL uncertain → WDO volatile
- Flattening: market expects stable/lower rates → BRL strengthening → WDO falls
- Inversion: recession signal → risk-off → BRL weakens → WDO rises
- Parallel shift up: hawkish surprise → BRL strengthens short-term → WDO falls
- Parallel shift down: dovish surprise → BRL weakens → WDO rises
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from takata_engine.pricing.di_curve import DICurve, DIPoint, MONTH_CODES

logger = logging.getLogger(__name__)

# Reverse month codes for parsing DI tickers
CODE_TO_MONTH = {v: k for k, v in MONTH_CODES.items()}

# Approximate business days per month (Brazil)
# More precise: use QuantLib or a proper bus-day calendar
AVG_BUS_DAYS_PER_MONTH = 21


def _parse_di_ticker(ticker: str, today: date) -> Optional[Tuple[date, int]]:
    """Parse a DI ticker into maturity date and approximate business days.

    B3 DI tickers: DI1FUT (generic), DI1F26 (Jan 2026), DI1N26 (Jul 2026), etc.
    Format: DI1{MonthCode}{YY} where month code follows B3 convention.

    Returns (maturity_date, business_days) or None if unparseable.
    """
    # Strip "DI1" prefix and "FUT" suffix
    clean = ticker.replace("DI1", "").replace("FUT", "").strip()
    if not clean or len(clean) < 2:
        return None

    month_code = clean[0].upper()
    year_str = clean[1:]

    if month_code not in CODE_TO_MONTH:
        return None

    try:
        month = CODE_TO_MONTH[month_code]
        year = int(year_str)
        # Handle 2-digit years
        if year < 100:
            year += 2000

        # DI futures expire on first business day of the month
        mat = date(year, month, 1)

        # Business days from today to maturity (approximate)
        cal_days = (mat - today).days
        if cal_days <= 0:
            return None

        # Approximate bus days: ~252/365 * cal_days
        bus_days = max(1, int(cal_days * 252 / 365))

        return mat, bus_days
    except (ValueError, OverflowError):
        return None


class LiveDICurve:
    """Full DI curve built from live Profit Ultra data.

    Fetches all DI tenors from the bridge, constructs the curve,
    and provides curve shape analysis for WDO signal generation.
    """

    def __init__(self, bridge_url: str = "http://127.0.0.1:8002"):
        self.bridge_url = bridge_url
        self.curve: Optional[DICurve] = None
        self._last_fetch: float = 0
        self._raw_contracts: Dict[str, Dict] = {}
        self._shape: Dict[str, Any] = {}
        # Slope dynamics tracking (rolling history of shape snapshots)
        self._slope_history: deque = deque(maxlen=600)  # ~5hrs at 30s intervals
        self._rate_history: Dict[str, deque] = {}  # per-tenor rate history
        self._daily_reset: str = ""

    def fetch(self, timeout: float = 3.0) -> bool:
        """Fetch all DI contracts from bridge and build curve."""
        try:
            r = requests.get(f"{self.bridge_url}/di_curve", timeout=timeout)
            data = r.json()
        except Exception as e:
            logger.debug("DI curve fetch failed: %s", e)
            return False

        contracts = data.get("contracts", {})
        if not contracts:
            # Fallback: try single DI1FUT quote
            try:
                r = requests.get(f"{self.bridge_url}/quote/DI1FUT", timeout=timeout)
                q = r.json()
                if q.get("last", 0) > 0:
                    contracts = {"DI1FUT": q}
            except Exception:
                return False

        self._raw_contracts = contracts
        today = date.today()
        points = []

        for ticker, q in contracts.items():
            rate = q.get("last", 0)
            if not rate or rate <= 0:
                continue

            parsed = _parse_di_ticker(ticker, today)
            if parsed is None:
                # Generic DI1FUT — assume front month (~126 bus days)
                if "FUT" in ticker:
                    points.append(DIPoint(
                        ticker=ticker, maturity=date(today.year, today.month + 6 if today.month <= 6 else today.month - 6, 1),
                        business_days=126, calendar_days=180, rate=float(rate),
                    ))
                continue

            mat, bus_days = parsed
            cal_days = (mat - today).days

            points.append(DIPoint(
                ticker=ticker, maturity=mat,
                business_days=bus_days, calendar_days=cal_days,
                rate=float(rate),
            ))

        if not points:
            return False

        self.curve = DICurve(today, points)
        self._last_fetch = time.time()
        self._analyze_shape()
        self._record_snapshot()

        logger.info("DI curve loaded: %d tenors, short=%.2f%%, long=%.2f%%",
                     len(points),
                     points[0].rate if points else 0,
                     points[-1].rate if points else 0)
        return True

    def needs_fetch(self, interval: int = 30) -> bool:
        return time.time() - self._last_fetch > interval

    # ── Curve Shape Analysis ──

    def _analyze_shape(self) -> None:
        """Analyze curve shape: slope, steepness, curvature, changes."""
        if not self.curve or len(self.curve.points) < 2:
            self._shape = {"shape": "insufficient_data", "slope_bps": 0}
            return

        pts = self.curve.points
        short_rate = pts[0].rate   # front end
        long_rate = pts[-1].rate   # long end

        # Find a mid-point if we have enough tenors
        mid_idx = len(pts) // 2
        mid_rate = pts[mid_idx].rate

        slope_bps = (long_rate - short_rate) * 100  # in basis points
        belly_bps = (mid_rate - (short_rate + long_rate) / 2) * 100  # curvature

        # Shape classification
        if slope_bps > 50:
            shape = "steep"        # long rates much higher than short
        elif slope_bps > 10:
            shape = "normal"       # healthy upward slope
        elif slope_bps > -10:
            shape = "flat"         # minimal difference
        elif slope_bps > -50:
            shape = "inverted"     # short rates higher than long
        else:
            shape = "deeply_inverted"  # extreme inversion

        # Butterfly: positive belly = hump, negative = dip
        butterfly = "neutral"
        if len(pts) >= 3:
            if belly_bps > 20:
                butterfly = "humped"   # belly rates higher than interpolation
            elif belly_bps < -20:
                butterfly = "concave"  # belly rates lower

        self._shape = {
            "shape": shape,
            "slope_bps": round(slope_bps, 1),
            "belly_bps": round(belly_bps, 1),
            "butterfly": butterfly,
            "short_rate": round(short_rate, 2),
            "mid_rate": round(mid_rate, 2),
            "long_rate": round(long_rate, 2),
            "short_tenor": pts[0].ticker,
            "long_tenor": pts[-1].ticker,
            "num_tenors": len(pts),
        }

    @property
    def shape(self) -> Dict[str, Any]:
        return self._shape

    # ── WDO-Specific Methods ──

    def wdo_fair_value(
        self,
        spot_usd: float,
        wdo_expiry_du: int = 21,
        dol_fut: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Compute WDO fair value by interpolating to the exact WDO expiry.

        WDO is always front-month. The DI rate at the WDO expiry tenor
        determines the theoretical fair value via covered interest rate parity.

        Parameters
        ----------
        spot_usd : float
            Current spot USDBRL (e.g. 5.19).
        wdo_expiry_du : int
            Business days to WDO expiry (typically 1-21 for front month).
        dol_fut : float, optional
            Current dollar futures price (B3 quotation, e.g. 5003.0). When
            provided, derives the live cupom cambial via BNDF1. The rolling
            1h mean of live cupom becomes the reference anchor for
            theoretical-price computation, replacing the crude
            ``di_at_expiry - 2.0`` heuristic. Divergence of live cupom from
            reference (``cupom_spread_bps``) is the EM stress signal input.
        """
        if not self.curve:
            return None

        # Interpolate DI rate to WDO expiry
        di_at_expiry = self.curve.interpolate(wdo_expiry_du)

        # Also get a longer tenor (3M) for forward rate
        di_3m = self.curve.interpolate(63)  # ~3 months

        # Forward rate from now to WDO expiry
        fwd_rate = self.curve.forward_rate(1, wdo_expiry_du) if wdo_expiry_du > 1 else di_at_expiry

        cal_days = wdo_expiry_du * 365 / 252
        onshore_factor = (1 + di_at_expiry / 100) ** (wdo_expiry_du / 252)

        # Live cupom from BNDF1 when dol_fut available.
        # cupom_cambial() already returns the rate as a percentage (e.g. 8.5).
        cupom_live: Optional[float] = None
        if dol_fut and dol_fut > 0:
            try:
                cupom_live = self.curve.cupom_cambial(wdo_expiry_du, spot_usd, dol_fut)
            except Exception as e:
                logger.debug("cupom_cambial failed: %s", e)
                cupom_live = None

        # Rolling 1h reference anchor (per-instance). Fallback = di - 2.0.
        if not hasattr(self, "_cupom_hist"):
            self._cupom_hist: deque = deque(maxlen=1800)  # ~1h at 2s polling
        if cupom_live is not None:
            self._cupom_hist.append((time.time(), cupom_live))

        now = time.time()
        recent = [c for (t, c) in self._cupom_hist if now - t <= 3600]
        if recent:
            cupom_reference = sum(recent) / len(recent)
        else:
            cupom_reference = di_at_expiry - 2.0  # warmup fallback

        offshore_factor = 1 + (cupom_reference / 100) * cal_days / 360
        theoretical = spot_usd * 1000 * onshore_factor / offshore_factor

        spread_bps: Optional[float] = None
        if cupom_live is not None:
            spread_bps = round((cupom_live - cupom_reference) * 100, 1)

        return {
            "theoretical": round(theoretical, 2),
            "di_at_expiry": round(di_at_expiry, 4),
            "di_3m": round(di_3m, 4),
            "forward_rate": round(fwd_rate, 4),
            "onshore_factor": round(onshore_factor, 6),
            "cupom_approx": round(cupom_reference, 2),   # back-compat alias
            "cupom_live": None if cupom_live is None else round(cupom_live, 4),
            "cupom_reference": round(cupom_reference, 4),
            "cupom_spread_bps": spread_bps,
            "wdo_expiry_du": wdo_expiry_du,
        }

    # ── Slope Dynamics ──

    def _record_snapshot(self) -> None:
        """Record current curve state for dynamics tracking."""
        today = date.today().isoformat()
        if self._daily_reset != today:
            self._slope_history.clear()
            self._rate_history.clear()
            self._daily_reset = today

        if not self._shape or self._shape.get("shape") == "insufficient_data":
            return

        self._slope_history.append({
            "ts": time.time(),
            "slope_bps": self._shape["slope_bps"],
            "short_rate": self._shape["short_rate"],
            "mid_rate": self._shape.get("mid_rate", 0),
            "long_rate": self._shape["long_rate"],
            "belly_bps": self._shape.get("belly_bps", 0),
            "shape": self._shape["shape"],
        })

        # Per-tenor rate history
        if self.curve:
            for p in self.curve.points:
                if p.ticker not in self._rate_history:
                    self._rate_history[p.ticker] = deque(maxlen=600)
                self._rate_history[p.ticker].append((time.time(), p.rate))

    def slope_dynamics(self) -> Dict[str, Any]:
        """Analyze slope changes over time — steepening, flattening, shifts.

        Returns dict with:
        - ``slope_change_5m``: slope change in bps over last 5 min
        - ``slope_change_30m``: slope change in bps over last 30 min
        - ``slope_velocity``: rate of slope change (bps/min)
        - ``movement``: steepening / flattening / stable
        - ``parallel_shift``: detected parallel shift (all tenors moving together)
        - ``shift_direction``: up / down / none
        - ``shift_magnitude_bps``: how much the curve shifted
        - ``front_rate_momentum``: short rate change (most WDO-relevant)
        - ``score_adjustment``: additive signal score adjustment
        - ``reasons``: list of DI curve dynamic reasons
        """
        result: Dict[str, Any] = {
            "slope_change_5m": None,
            "slope_change_30m": None,
            "slope_velocity": None,
            "movement": "stable",
            "parallel_shift": False,
            "shift_direction": "none",
            "shift_magnitude_bps": 0.0,
            "front_rate_momentum": None,
            "score_adjustment": 0.0,
            "reasons": [],
        }

        if len(self._slope_history) < 2:
            return result

        reasons = result["reasons"]
        adj = 0.0
        now = time.time()

        # ── Slope change over 5m and 30m ──
        current = self._slope_history[-1]
        slope_5m = self._get_slope_at(now - 300)
        slope_30m = self._get_slope_at(now - 1800)

        if slope_5m is not None:
            d5 = current["slope_bps"] - slope_5m
            result["slope_change_5m"] = round(d5, 1)

            # Fast slope change = significant
            if abs(d5) > 10:
                if d5 > 0:
                    reasons.append("di_steepening_fast")
                    adj += 0.06  # steepening = rates rising long end = BRL pressure
                else:
                    reasons.append("di_flattening_fast")
                    adj += 0.06  # flattening = easing expected = BRL support

        if slope_30m is not None:
            d30 = current["slope_bps"] - slope_30m
            result["slope_change_30m"] = round(d30, 1)

            # Determine movement
            if d30 > 5:
                result["movement"] = "steepening"
            elif d30 < -5:
                result["movement"] = "flattening"

            # Velocity = bps/min
            mins = 30.0
            velocity = d30 / mins
            result["slope_velocity"] = round(velocity, 3)

        # ── Parallel shift detection ──
        shift_data = self._detect_parallel_shift(300)  # 5 min window
        if shift_data:
            result["parallel_shift"] = shift_data["is_parallel"]
            result["shift_direction"] = shift_data["direction"]
            result["shift_magnitude_bps"] = round(shift_data["magnitude_bps"], 1)

            if shift_data["is_parallel"] and shift_data["magnitude_bps"] > 3:
                if shift_data["direction"] == "up":
                    reasons.append("di_parallel_shift_up")
                    adj += 0.08  # hawkish surprise → BRL short-term strength → WDO down
                elif shift_data["direction"] == "down":
                    reasons.append("di_parallel_shift_down")
                    adj += 0.08  # dovish surprise → BRL weakness → WDO up

        # ── Front rate momentum (most relevant for WDO) ──
        front_mom = self._front_rate_momentum(300)
        if front_mom is not None:
            result["front_rate_momentum"] = round(front_mom, 3)
            if abs(front_mom) > 0.05:
                if front_mom > 0:
                    reasons.append("di_front_rising")
                    adj += 0.05  # short rate spiking = hawkish
                else:
                    reasons.append("di_front_falling")
                    adj += 0.05  # short rate dropping = dovish

        result["score_adjustment"] = round(adj, 3)
        return result

    def _get_slope_at(self, target_ts: float) -> Optional[float]:
        """Get the slope_bps closest to a target timestamp."""
        if not self._slope_history:
            return None
        best = None
        best_dist = float("inf")
        for snap in self._slope_history:
            dist = abs(snap["ts"] - target_ts)
            if dist < best_dist:
                best_dist = dist
                best = snap["slope_bps"]
        # Only use if within 60s of target
        return best if best_dist < 60 else None

    def _detect_parallel_shift(self, lookback_seconds: int) -> Optional[Dict[str, Any]]:
        """Detect if all tenors moved in the same direction (parallel shift).

        A parallel shift means the entire curve moved up or down together,
        as opposed to steepening/flattening where only parts move.
        """
        if not self._rate_history or len(self._rate_history) < 2:
            return None

        now = time.time()
        cutoff = now - lookback_seconds

        changes = []
        for ticker, hist in self._rate_history.items():
            if len(hist) < 2:
                continue
            # Find rate at cutoff
            old_rate = None
            for ts, rate in hist:
                if ts >= cutoff:
                    old_rate = rate
                    break
            if old_rate is None:
                continue
            current_rate = hist[-1][1]
            change_bps = (current_rate - old_rate) * 100
            changes.append(change_bps)

        if len(changes) < 2:
            return None

        # Parallel = all changes same sign and similar magnitude
        avg_change = sum(changes) / len(changes)
        all_positive = all(c > 0.5 for c in changes)
        all_negative = all(c < -0.5 for c in changes)
        is_parallel = all_positive or all_negative

        # Check dispersion (std dev of changes)
        std = float((sum((c - avg_change) ** 2 for c in changes) / len(changes)) ** 0.5)
        # Low dispersion relative to magnitude = parallel
        if abs(avg_change) > 0 and std / abs(avg_change) > 0.5:
            is_parallel = False  # too dispersed, not a clean parallel shift

        direction = "up" if avg_change > 0.5 else "down" if avg_change < -0.5 else "none"

        return {
            "is_parallel": is_parallel,
            "direction": direction,
            "magnitude_bps": abs(avg_change),
            "dispersion_bps": round(std, 1),
            "tenors_used": len(changes),
        }

    def _front_rate_momentum(self, lookback_seconds: int) -> Optional[float]:
        """Change in the front (shortest) DI rate over lookback period."""
        if not self._slope_history or len(self._slope_history) < 2:
            return None

        now = time.time()
        cutoff = now - lookback_seconds

        old_rate = None
        for snap in self._slope_history:
            if snap["ts"] >= cutoff:
                old_rate = snap["short_rate"]
                break

        if old_rate is None:
            return None

        return self._slope_history[-1]["short_rate"] - old_rate

    def curve_signal(self) -> Dict[str, Any]:
        """Generate WDO trading signal from curve shape and dynamics.

        Returns signal data including direction bias and confidence.
        """
        s = self._shape
        if not s or s.get("shape") == "insufficient_data":
            return {"signal": "neutral", "confidence": 0, "reason": "no curve data"}

        signal = "neutral"
        confidence = 0.0
        reasons = []

        shape = s["shape"]
        slope = s["slope_bps"]

        # Curve shape → WDO directional bias
        if shape == "steep":
            # Steep = market expects higher rates = BRL could weaken = WDO up bias
            # BUT also: steep = carry attractive = foreign flows = BRL strengthens
            # Ambiguous — low confidence
            signal = "neutral"
            confidence = 0.3
            reasons.append("steep_curve_ambiguous")
        elif shape == "inverted" or shape == "deeply_inverted":
            # Inversion = recession fears = risk-off = BRL weakens = WDO rises
            signal = "wdo_up"
            confidence = 0.7
            reasons.append("inverted_curve_risk_off")
        elif shape == "flat":
            # Flat = uncertainty, low conviction
            signal = "neutral"
            confidence = 0.2
            reasons.append("flat_curve_uncertain")
        elif shape == "normal":
            # Normal healthy curve = stable environment = BRL supported = WDO stable/down
            signal = "wdo_down"
            confidence = 0.4
            reasons.append("normal_curve_brl_stable")

        # Butterfly adds nuance
        if s.get("butterfly") == "humped":
            reasons.append("belly_humped_policy_uncertainty")
            confidence *= 0.8  # reduce confidence
        elif s.get("butterfly") == "concave":
            reasons.append("belly_concave_easing_expected")

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "reasons": reasons,
            "shape": shape,
            "slope_bps": slope,
            "short_rate": s.get("short_rate"),
            "long_rate": s.get("long_rate"),
        }

    # ── Serialization ──

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loaded": self.curve is not None,
            "num_tenors": len(self.curve.points) if self.curve else 0,
            "shape": self._shape,
            "dynamics": self.slope_dynamics(),
            "points": [
                {"ticker": p.ticker, "du": p.business_days, "rate": p.rate, "fra": round(p.fra, 4)}
                for p in (self.curve.points if self.curve else [])
            ],
            "last_fetch": self._last_fetch,
            "history_depth": len(self._slope_history),
        }
