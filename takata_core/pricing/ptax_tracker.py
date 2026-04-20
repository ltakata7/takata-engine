"""PTAX Microstructure Tracker — intraday PTAX formation analysis for WDO.

Tracks the PTAX fixing process across BCB's 4 consultation windows:
- Window 1: 10:00-10:10 BRT
- Window 2: 11:00-11:10 BRT
- Window 3: 12:00-12:10 BRT
- Window 4: 13:00-13:10 BRT (final → PTAX published)

Key signals:
- **Window proximity**: countdown to next window → flow anticipation
- **Partial PTAX formation**: VWAP during each window approximates BCB's sample
- **PTAX vs spot divergence**: WDO drifting from forming PTAX → mean reversion
- **Window transition momentum**: flow direction change between windows
- **Post-PTAX release**: convergence/divergence after 13:10

WDO is priced off PTAX expectations. During formation (10:00-13:10),
order flow around the windows drives 60%+ of intraday variance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)

BRT = pytz.timezone("America/Sao_Paulo")

# BCB consultation windows (BRT)
PTAX_WINDOWS = [
    {"id": 1, "start_h": 10, "start_m": 0, "end_h": 10, "end_m": 10},
    {"id": 2, "start_h": 11, "start_m": 0, "end_h": 11, "end_m": 10},
    {"id": 3, "start_h": 12, "start_m": 0, "end_h": 12, "end_m": 10},
    {"id": 4, "start_h": 13, "start_m": 0, "end_h": 13, "end_m": 10},
]


@dataclass
class WindowSnapshot:
    """Price/flow snapshot captured during a PTAX window."""
    window_id: int
    vwap: float = 0.0          # VWAP during the window
    price_open: float = 0.0    # price at window open
    price_close: float = 0.0   # price at window close
    agg_balance: float = 0.0   # net aggression during window
    high: float = 0.0
    low: float = 0.0
    captured: bool = False


@dataclass
class PTAXState:
    """Intraday PTAX formation state — resets each day."""
    date: str = ""
    windows: Dict[int, WindowSnapshot] = field(default_factory=dict)
    # Running estimate of PTAX based on window VWAPs
    ptax_estimate: float = 0.0
    # Previous day's PTAX (from BCB or yesterday's estimate)
    ptax_yesterday: float = 0.0
    # Tracks last known price at each window transition
    window_transition_prices: List[float] = field(default_factory=list)

    def reset(self, today: str) -> None:
        self.date = today
        self.windows = {i: WindowSnapshot(window_id=i) for i in range(1, 5)}
        self.ptax_estimate = 0.0
        self.window_transition_prices = []


class PTAXTracker:
    """Tracks PTAX formation intraday and generates microstructure signals.

    Call ``update()`` every scan cycle (~2s) with current price + tape data.
    Call ``signal()`` to get the current PTAX microstructure signal dict.
    """

    def __init__(self) -> None:
        self._state = PTAXState()
        self._last_window_id: int = 0  # which window we were in last tick
        self._pre_window_prices: Dict[int, float] = {}  # price 60s before each window

    def update(
        self,
        price: float,
        vwap: float,
        agg_balance: float,
        high: float = 0.0,
        low: float = 0.0,
        ptax_yesterday: float = 0.0,
    ) -> None:
        """Update PTAX state with current market data. Call every scan cycle."""
        now = datetime.now(BRT)
        today_str = now.strftime("%Y-%m-%d")

        # Day reset
        if self._state.date != today_str:
            self._state.reset(today_str)
            self._last_window_id = 0
            self._pre_window_prices = {}

        if ptax_yesterday > 0:
            self._state.ptax_yesterday = ptax_yesterday

        h, m = now.hour, now.minute
        current_window = self._get_current_window(h, m)

        # Capture pre-window price (1 min before window opens)
        for w in PTAX_WINDOWS:
            wid = w["id"]
            pre_h, pre_m = w["start_h"], w["start_m"] - 1
            if pre_m < 0:
                pre_h -= 1
                pre_m = 59
            if h == pre_h and m == pre_m and wid not in self._pre_window_prices:
                self._pre_window_prices[wid] = price

        # If inside a window, accumulate data
        if current_window:
            snap = self._state.windows[current_window]
            if not snap.captured:
                # First tick of this window
                snap.price_open = price
                snap.high = price
                snap.low = price
            snap.high = max(snap.high, price)
            snap.low = min(snap.low, price)
            snap.vwap = vwap  # bridge VWAP is session-wide, good proxy
            snap.agg_balance = agg_balance
            snap.price_close = price
            snap.captured = True

        # Detect window transition (exiting a window)
        if self._last_window_id > 0 and current_window != self._last_window_id:
            prev = self._state.windows.get(self._last_window_id)
            if prev and prev.captured:
                self._state.window_transition_prices.append(prev.price_close)

        self._last_window_id = current_window or self._last_window_id

        # Update running PTAX estimate from captured window VWAPs
        self._update_ptax_estimate(price)

    def signal(self, current_price: float) -> Dict[str, Any]:
        """Generate PTAX microstructure signal for the WDO scanner.

        Returns dict with:
        - ``proximity``: seconds until next window, urgency level
        - ``ptax_estimate``: running PTAX estimate from window data
        - ``ptax_vs_spot``: divergence of WDO from forming PTAX (pts)
        - ``window_momentum``: flow direction change between windows
        - ``signal``: overall PTAX signal (convergence/divergence/neutral)
        - ``score_adjustment``: additive adjustment to signal score
        - ``reasons``: list of signal reasons
        """
        now = datetime.now(BRT)
        h, m = now.hour, now.minute
        result: Dict[str, Any] = {
            "proximity": self._proximity_signal(h, m),
            "windows_captured": sum(1 for s in self._state.windows.values() if s.captured),
            "ptax_estimate": round(self._state.ptax_estimate, 2) if self._state.ptax_estimate else 0,
            "ptax_yesterday": round(self._state.ptax_yesterday, 2) if self._state.ptax_yesterday else 0,
            "signal": "neutral",
            "score_adjustment": 0.0,
            "reasons": [],
        }

        reasons = result["reasons"]
        adj = 0.0

        # 1. PTAX vs spot divergence (only meaningful during/after formation)
        if self._state.ptax_estimate > 0 and current_price > 0:
            div_pts = current_price - self._state.ptax_estimate
            result["ptax_vs_spot"] = round(div_pts, 1)

            # Large divergence = mean reversion opportunity
            if abs(div_pts) > 8:
                reasons.append("ptax_large_divergence")
                adj += 0.10
                result["signal"] = "convergence"
            elif abs(div_pts) > 4:
                reasons.append("ptax_moderate_divergence")
                adj += 0.06

        # 2. Window momentum — flow direction changing between windows
        momentum = self._window_momentum()
        result["window_momentum"] = momentum
        if momentum["consistent"]:
            reasons.append("ptax_flow_consistent")
            adj += 0.08  # consistent flow across windows = strong signal
        elif momentum["reversal"]:
            reasons.append("ptax_flow_reversal")
            adj -= 0.05  # flow reversed between windows = uncertainty

        # 3. Window range expansion — wide windows = volatile PTAX formation
        range_data = self._window_range_analysis()
        result["window_ranges"] = range_data
        if range_data.get("expanding"):
            reasons.append("ptax_range_expanding")
            adj -= 0.04  # widening ranges = more uncertainty

        # 4. PTAX vs yesterday — directional bias from PTAX change
        if self._state.ptax_estimate > 0 and self._state.ptax_yesterday > 0:
            ptax_change = self._state.ptax_estimate - self._state.ptax_yesterday
            result["ptax_change_from_yesterday"] = round(ptax_change, 2)
            if abs(ptax_change) > 20:
                reasons.append("ptax_big_move")
                adj += 0.06
            result["ptax_direction"] = "up" if ptax_change > 0 else "down" if ptax_change < 0 else "flat"

        # 5. Pre-window positioning — price moved before window (anticipation)
        anticipation = self._anticipation_signal(current_price, h, m)
        if anticipation:
            result["anticipation"] = anticipation
            if anticipation.get("strong"):
                reasons.append("ptax_anticipation")
                adj += 0.05

        result["score_adjustment"] = round(adj, 3)
        return result

    # ── Internal helpers ──

    def _get_current_window(self, h: int, m: int) -> Optional[int]:
        for w in PTAX_WINDOWS:
            if w["start_h"] == h and w["start_m"] <= m <= w["end_m"]:
                return w["id"]
            # Handle edge: window 4 is 13:00-13:10
            if w["start_h"] == h and m >= w["start_m"] and m <= w["end_m"]:
                return w["id"]
        return None

    def _proximity_signal(self, h: int, m: int) -> Dict[str, Any]:
        """How close are we to the next PTAX window?"""
        now_mins = h * 60 + m

        for w in PTAX_WINDOWS:
            start_mins = w["start_h"] * 60 + w["start_m"]
            end_mins = w["end_h"] * 60 + w["end_m"]

            # Currently in a window
            if start_mins <= now_mins <= end_mins:
                remaining = (end_mins - now_mins) * 60
                return {
                    "in_window": True,
                    "window_id": w["id"],
                    "seconds_remaining": remaining,
                    "urgency": "active",
                }

            # Before the window
            if now_mins < start_mins:
                seconds_until = (start_mins - now_mins) * 60
                urgency = "imminent" if seconds_until <= 120 else "approaching" if seconds_until <= 600 else "distant"
                return {
                    "in_window": False,
                    "next_window": w["id"],
                    "seconds_until": seconds_until,
                    "urgency": urgency,
                }

        # After all windows
        return {"in_window": False, "next_window": None, "urgency": "post_ptax"}

    def _update_ptax_estimate(self, current_price: float) -> None:
        """Running PTAX estimate from captured window data.

        BCB's PTAX is a weighted average of dealer quotes during windows.
        We approximate using VWAP snapshots from each captured window.
        """
        captured = [s for s in self._state.windows.values() if s.captured and s.vwap > 0]
        if not captured:
            # Before any windows: use current VWAP as estimate
            if current_price > 0:
                self._state.ptax_estimate = current_price
            return

        # Weight later windows more (BCB weights final window heavily)
        weights = {1: 0.15, 2: 0.20, 3: 0.25, 4: 0.40}
        total_weight = sum(weights.get(s.window_id, 0.25) for s in captured)
        estimate = sum(s.vwap * weights.get(s.window_id, 0.25) for s in captured) / total_weight
        self._state.ptax_estimate = estimate

    def _window_momentum(self) -> Dict[str, Any]:
        """Analyze flow direction changes between consecutive windows."""
        captured = sorted(
            [s for s in self._state.windows.values() if s.captured],
            key=lambda s: s.window_id,
        )
        if len(captured) < 2:
            return {"consistent": False, "reversal": False, "direction": "unknown"}

        # Compare aggression balance across windows
        directions = []
        for s in captured:
            if s.agg_balance > 200:
                directions.append(1)
            elif s.agg_balance < -200:
                directions.append(-1)
            else:
                directions.append(0)

        non_zero = [d for d in directions if d != 0]
        if not non_zero:
            return {"consistent": False, "reversal": False, "direction": "neutral"}

        consistent = all(d == non_zero[0] for d in non_zero)

        # Check for reversal (last direction different from first)
        reversal = len(non_zero) >= 2 and non_zero[-1] != non_zero[0]

        direction = "buy" if non_zero[-1] > 0 else "sell" if non_zero[-1] < 0 else "neutral"

        return {
            "consistent": consistent,
            "reversal": reversal,
            "direction": direction,
            "flow_sequence": directions,
        }

    def _window_range_analysis(self) -> Dict[str, Any]:
        """Analyze range expansion/contraction across windows."""
        captured = sorted(
            [s for s in self._state.windows.values() if s.captured and s.high > s.low],
            key=lambda s: s.window_id,
        )
        if len(captured) < 2:
            return {"expanding": False, "ranges": []}

        ranges = [s.high - s.low for s in captured]
        expanding = ranges[-1] > ranges[0] * 1.3  # 30% wider than first window
        return {
            "expanding": expanding,
            "ranges": [round(r, 1) for r in ranges],
            "avg_range": round(sum(ranges) / len(ranges), 1),
        }

    def _anticipation_signal(self, current_price: float, h: int, m: int) -> Optional[Dict[str, Any]]:
        """Detect if price moved significantly before a window (anticipation)."""
        now_mins = h * 60 + m

        for w in PTAX_WINDOWS:
            wid = w["id"]
            start_mins = w["start_h"] * 60 + w["start_m"]
            # 5 minutes before window
            if 0 < (start_mins - now_mins) <= 5 and wid in self._pre_window_prices:
                pre_price = self._pre_window_prices[wid]
                move = current_price - pre_price
                if abs(move) > 3:  # > 3 pts in last minute = anticipation
                    return {
                        "window_id": wid,
                        "move_pts": round(move, 1),
                        "direction": "up" if move > 0 else "down",
                        "strong": abs(move) > 6,
                    }
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API/frontend."""
        return {
            "date": self._state.date,
            "ptax_estimate": round(self._state.ptax_estimate, 2) if self._state.ptax_estimate else 0,
            "ptax_yesterday": round(self._state.ptax_yesterday, 2) if self._state.ptax_yesterday else 0,
            "windows": {
                wid: {
                    "captured": s.captured,
                    "vwap": round(s.vwap, 2) if s.vwap else 0,
                    "open": round(s.price_open, 2),
                    "close": round(s.price_close, 2),
                    "high": round(s.high, 2),
                    "low": round(s.low, 2),
                    "range": round(s.high - s.low, 1) if s.high > s.low else 0,
                    "flow": round(s.agg_balance, 0),
                }
                for wid, s in self._state.windows.items()
            },
            "transitions": [round(p, 2) for p in self._state.window_transition_prices],
        }
