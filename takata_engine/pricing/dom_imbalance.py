"""DOM (Depth of Market) imbalance — Level 2 bid/ask ladder analysis.

The top N levels of bid + ask ladder reveal the *resting* liquidity at each
price. When one side has dramatically more size than the other, it's a
short-term directional pull:

- **Bid stack ≫ ask stack** = buyers parked thick orders, sellers thin
  → bullish pull. Aggressors hitting the offer face less resistance going up,
  defenders sitting on the bid soak any seller pressure.
- **Ask stack ≫ bid stack** = bearish pull, opposite.

Three levels of refinement:

1. **Top-of-book ratio** (level 1 only): bid_size / (bid_size + ask_size)
2. **Top-N stack ratio**: same, summed over top N levels
3. **Persistence**: imbalance has to hold for N consecutive ticks to count —
   a single fat order that pulls in 200ms is noise, not signal

Generates signals:
- ``dom_imbalance_bull``  — bid stack ≥ threshold × ask stack, persistent
- ``dom_imbalance_bear``  — ask stack ≥ threshold × bid stack, persistent
- ``dom_thin_book``       — both sides extremely thin (low conviction)

Requires ``reqMktDepth`` subscription (CME L2 entitlement). The aggregator
delivers per-level updates via the standard pendingTickersEvent channel.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default top-N levels to aggregate. CME L2 typically delivers 10 levels;
# the top 5 carry the actionable resting size. Beyond 5 is mostly noise.
TOP_N_LEVELS = 5

# How many consecutive ticks the imbalance must persist before firing.
# 1-2 = too noisy (random spikes from a single fat order); 5+ = sluggish.
PERSISTENCE_TICKS = 3

# Bull/bear threshold: top side / opposite side ≥ this → imbalance.
IMBALANCE_THRESHOLD = 1.8

# "Thin book" flag — both sides combined size below this many contracts.
# Means the book is genuinely empty, not just imbalanced.
THIN_BOOK_TOTAL = 30

# Rolling window for `bid_size_history` / `ask_size_history` (ticks).
HISTORY_LEN = 60


@dataclass
class DOMState:
    bid_levels: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    ask_levels: List[Tuple[float, float]] = field(default_factory=list)
    last_update_ts: float = 0.0
    bid_size_history: Deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    ask_size_history: Deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bid_levels": [[round(p, 4), round(s, 0)] for p, s in self.bid_levels],
            "ask_levels": [[round(p, 4), round(s, 0)] for p, s in self.ask_levels],
            "last_update_ts": self.last_update_ts,
            "last_update_iso": datetime.fromtimestamp(self.last_update_ts).isoformat() if self.last_update_ts else None,
        }


class DOMImbalanceTracker:
    """Tracks bid/ask ladder, computes imbalance ratios with persistence."""

    def __init__(self, top_n: int = TOP_N_LEVELS,
                 persistence_ticks: int = PERSISTENCE_TICKS,
                 imbalance_threshold: float = IMBALANCE_THRESHOLD) -> None:
        self.top_n = top_n
        self.persistence_ticks = persistence_ticks
        self.imbalance_threshold = imbalance_threshold
        self._state = DOMState()
        # Streak: which direction has the imbalance held for, and for how long.
        self._streak_dir: str = "neutral"
        self._streak_len: int = 0

    def update_level(self, side: str, position: int, price: float, size: float) -> None:
        """Update a single level of the book.

        IBKR delivers ladder updates as (side, position, price, size, operation).
        We accept the simplified form here; the IB-side handler in
        tick_aggregator owns the operation translation.

        Parameters
        ----------
        side : "bid" | "ask"
        position : int
            0-indexed level (0 = top of book).
        price : float
        size : float
        """
        ts = datetime.now().timestamp()
        levels = self._state.bid_levels if side == "bid" else self._state.ask_levels
        # Grow the list as needed.
        while len(levels) <= position:
            levels.append((0.0, 0.0))
        if size <= 0:
            # Level removed — clear it.
            levels[position] = (0.0, 0.0)
        else:
            levels[position] = (float(price), float(size))
        self._state.last_update_ts = ts
        self._evaluate_streak()

    def replace_book(self, bid_levels: List[Tuple[float, float]],
                     ask_levels: List[Tuple[float, float]]) -> None:
        """Replace the entire book in one shot. Useful for tests + initial snapshot."""
        self._state.bid_levels = list(bid_levels)
        self._state.ask_levels = list(ask_levels)
        self._state.last_update_ts = datetime.now().timestamp()
        self._evaluate_streak()

    def _top_n_size(self, levels: List[Tuple[float, float]]) -> float:
        return sum(size for _, size in levels[:self.top_n] if size > 0)

    def _evaluate_streak(self) -> None:
        bid_total = self._top_n_size(self._state.bid_levels)
        ask_total = self._top_n_size(self._state.ask_levels)
        self._state.bid_size_history.append(bid_total)
        self._state.ask_size_history.append(ask_total)

        if bid_total <= 0 or ask_total <= 0:
            self._streak_dir = "neutral"
            self._streak_len = 0
            return

        # Determine this tick's direction.
        bid_ratio = bid_total / ask_total
        if bid_ratio >= self.imbalance_threshold:
            tick_dir = "bull"
        elif (1.0 / bid_ratio) >= self.imbalance_threshold:
            tick_dir = "bear"
        else:
            tick_dir = "neutral"

        if tick_dir == self._streak_dir:
            self._streak_len += 1
        else:
            self._streak_dir = tick_dir
            self._streak_len = 1 if tick_dir != "neutral" else 0

    def signal(self) -> Dict[str, Any]:
        """Return the current DOM imbalance signal for the scanner."""
        bid_total = self._top_n_size(self._state.bid_levels)
        ask_total = self._top_n_size(self._state.ask_levels)
        total = bid_total + ask_total
        ratio_b = bid_total / ask_total if ask_total > 0 else 0.0

        thin = total > 0 and total < THIN_BOOK_TOTAL
        active = (
            self._streak_dir != "neutral"
            and self._streak_len >= self.persistence_ticks
            and not thin
        )

        reasons: List[str] = []
        score_adjustment = 0.0
        if active:
            reasons.append(f"dom_imbalance_{self._streak_dir}")
            # Score scales with persistence; cap at 0.12 so DOM is a confirming
            # voice rather than a primary trigger.
            score_adjustment = min(0.12, 0.025 * self._streak_len)
        if thin:
            reasons.append("dom_thin_book")

        return {
            "active": active,
            "direction": self._streak_dir if active else "none",
            "persistence": self._streak_len,
            "bid_top_n_size": round(bid_total, 0),
            "ask_top_n_size": round(ask_total, 0),
            "ratio_bid_over_ask": round(ratio_b, 2),
            "thin": thin,
            "reasons": reasons,
            "score_adjustment": score_adjustment,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_n": self.top_n,
            "persistence_ticks_required": self.persistence_ticks,
            "imbalance_threshold": self.imbalance_threshold,
            "state": self._state.to_dict(),
            "signal": self.signal(),
        }

    def reset(self) -> None:
        self._state = DOMState()
        self._streak_dir = "neutral"
        self._streak_len = 0


_singletons: Dict[str, DOMImbalanceTracker] = {}


def get_dom_tracker(instrument: str = "MES") -> DOMImbalanceTracker:
    if instrument not in _singletons:
        _singletons[instrument] = DOMImbalanceTracker()
    return _singletons[instrument]


def reset_all_dom_trackers() -> None:
    for t in _singletons.values():
        t.reset()
    _singletons.clear()
