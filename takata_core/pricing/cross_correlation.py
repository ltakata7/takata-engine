"""Cross-asset correlation tracker for WDO fast-timeframe trading.

Maintains rolling Pearson correlation between WDO and four cross-asset
references: IND (IBOV futures), ES (S&P 500 mini), NQ (Nasdaq mini),
and BIT (Bitcoin futures). Detects when correlations break down —
which is often a stronger signal than the correlation itself.

**Why cross-correlation matters for WDO 3-5min trading:**
- WDO vs IND is normally strongly negative (BRL risk-on = IBOV up, WDO down).
  When the correlation collapses, flow is idiosyncratic — often Brazil-specific
  (political, EMBI, NDF flow, PTAX prep).
- WDO vs ES tracks global risk in US hours. ES up with WDO up = genuine
  dollar strength vs BRL (not risk appetite driving it).
- WDO vs NQ tracks tech-risk. Strong inverse during tech-led rallies.
- WDO vs BIT — crypto acts as a leading global risk proxy in off-hours.

Signal types generated:
- ``xcorr_ind_breakdown``   — WDO/IND correlation near zero or flipped
- ``xcorr_ind_extreme_neg`` — very strong inverse = classic risk-on/off day
- ``xcorr_es_divergence``   — WDO moving opposite to ES direction
- ``xcorr_global_aligned``  — ES + NQ + IND all moving same direction = consensus
- ``xcorr_brazil_isolated`` — all global pairs flat but WDO moving = idiosyncratic

Call ``update(wdo_price, {symbol: price})`` every scan cycle.
Call ``signal(wdo_price)`` for the correlation dict.
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

# Cross-asset tickers we track
CROSS_ASSETS = ("IND", "ES", "NQ", "BIT")

# Rolling correlation window: 150 ticks ≈ 5 min at 2s cadence
CORR_WINDOW = 150

# Historical correlation baseline (bootstrapped as we learn)
BASELINE_WINDOW = 600  # ~20 min

# Threshold for "breakdown" detection
CORR_BREAKDOWN_THRESHOLD = 0.30  # |corr| fell below this vs baseline
CORR_EXTREME_NEG = -0.75


@dataclass
class PairState:
    """Per-pair price buffer + correlation history."""
    wdo_prices: Deque[float] = field(default_factory=lambda: deque(maxlen=CORR_WINDOW))
    other_prices: Deque[float] = field(default_factory=lambda: deque(maxlen=CORR_WINDOW))
    # Historical correlations for baseline
    corr_history: Deque[float] = field(default_factory=lambda: deque(maxlen=BASELINE_WINDOW))


@dataclass
class CrossAssetState:
    """Full tracker state — one pair per cross-asset."""
    date: str = ""
    pairs: Dict[str, PairState] = field(default_factory=dict)

    def reset(self, today: str) -> None:
        self.date = today
        self.pairs = {sym: PairState() for sym in CROSS_ASSETS}


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient of two equal-length series.

    Returns 0 if series are too short or either has zero variance.
    """
    n = len(xs)
    if n < 10 or n != len(ys):
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return num / math.sqrt(var_x * var_y)


class CrossCorrelationTracker:
    """Rolling correlation tracker for WDO vs global/Brazil references."""

    def __init__(self) -> None:
        self._state = CrossAssetState()
        self._state.reset(datetime.now(BRT).strftime("%Y-%m-%d"))

    def update(self, wdo_price: float, other_prices: Dict[str, float]) -> None:
        """Update price buffers.

        Parameters
        ----------
        wdo_price : float
            Current WDO price.
        other_prices : dict
            Mapping of cross-asset ticker -> current price, e.g.
            ``{"IND": 130450, "ES": 5810.5, "NQ": 20420, "BIT": 98500}``.
            Missing keys are skipped (not all markets are open 24/7).
        """
        if wdo_price <= 0:
            return
        now = datetime.now(BRT)
        today_str = now.strftime("%Y-%m-%d")
        if self._state.date != today_str:
            self._state.reset(today_str)

        for sym in CROSS_ASSETS:
            if sym not in other_prices:
                continue
            p = other_prices[sym]
            if not p or p <= 0:
                continue
            pair = self._state.pairs[sym]
            pair.wdo_prices.append(wdo_price)
            pair.other_prices.append(p)

            # Record latest correlation if we have enough samples
            if len(pair.wdo_prices) >= 30:
                corr = _pearson(list(pair.wdo_prices), list(pair.other_prices))
                pair.corr_history.append(corr)

    def signal(self, wdo_price: float) -> Dict[str, Any]:
        """Compute correlation signal dict."""
        result: Dict[str, Any] = {
            "correlations": {},
            "baselines": {},
            "signal": "neutral",
            "score_adjustment": 0.0,
            "reasons": [],
        }
        reasons: List[str] = result["reasons"]
        adj = 0.0

        live_corrs: Dict[str, float] = {}
        baselines: Dict[str, float] = {}

        for sym, pair in self._state.pairs.items():
            if len(pair.wdo_prices) < 30:
                continue
            corr = _pearson(list(pair.wdo_prices), list(pair.other_prices))
            live_corrs[sym] = round(corr, 3)
            # Baseline = mean of older correlations (excluding most recent 30)
            if len(pair.corr_history) >= 100:
                hist = list(pair.corr_history)
                baseline = sum(hist[:-30]) / len(hist[:-30])
                baselines[sym] = round(baseline, 3)
            else:
                baselines[sym] = 0.0

        result["correlations"] = live_corrs
        result["baselines"] = baselines

        # ── IND (IBOV futures) — Brazil risk-on/off primary gauge ──
        ind_corr = live_corrs.get("IND")
        ind_base = baselines.get("IND", 0.0)
        if ind_corr is not None:
            if ind_corr <= CORR_EXTREME_NEG:
                reasons.append("xcorr_ind_extreme_neg")
                adj += 0.06  # textbook risk-on/off day = follow structural reads
                result["signal"] = "risk_regime_active"
            elif ind_base < -0.4 and abs(ind_corr) < CORR_BREAKDOWN_THRESHOLD:
                # Usually strong inverse, now ~0 = correlation collapse
                reasons.append("xcorr_ind_breakdown")
                adj += 0.07  # idiosyncratic WDO flow = PTAX/EMBI/local event
                result["signal"] = "brl_idiosyncratic"

        # ── ES — US risk proxy ──
        es_corr = live_corrs.get("ES")
        if es_corr is not None:
            # If WDO strongly disagrees with ES direction on short window
            if es_corr >= 0.5:
                # Positive ES/WDO correlation = dollar-strength driven
                # (both rising = global USD demand, not risk-off)
                reasons.append("xcorr_es_divergence")
                adj += 0.04

        # ── Global consensus: ES + NQ + IND all same-signed correlation ──
        signs = []
        for s in ("ES", "NQ", "IND"):
            c = live_corrs.get(s)
            if c is not None and abs(c) >= 0.35:
                signs.append(1 if c > 0 else -1)
        if len(signs) >= 2 and all(s == signs[0] for s in signs):
            reasons.append("xcorr_global_aligned")
            adj += 0.05
            result["signal"] = "global_consensus"

        # ── Brazil isolated — global pairs all near zero, but WDO moving ──
        global_flat = all(
            abs(live_corrs.get(s, 0)) < CORR_BREAKDOWN_THRESHOLD
            for s in ("ES", "NQ", "IND")
            if s in live_corrs
        )
        if global_flat and len(live_corrs) >= 3:
            # Check if WDO itself has been moving (look at buffer variance)
            wdo_series: List[float] = []
            for pair in self._state.pairs.values():
                if len(pair.wdo_prices) >= 30:
                    wdo_series = list(pair.wdo_prices)
                    break
            if wdo_series:
                wdo_mean = sum(wdo_series) / len(wdo_series)
                wdo_var = sum((x - wdo_mean) ** 2 for x in wdo_series) / len(wdo_series)
                wdo_std = math.sqrt(max(wdo_var, 0))
                if wdo_std >= 1.5:  # WDO moved at least 1.5 pts std over window
                    reasons.append("xcorr_brazil_isolated")
                    adj += 0.05
                    result["signal"] = "brazil_idiosyncratic"

        result["score_adjustment"] = round(adj, 3)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Dump state for API/frontend."""
        out: Dict[str, Any] = {"date": self._state.date, "pairs": {}}
        for sym, pair in self._state.pairs.items():
            out["pairs"][sym] = {
                "samples": len(pair.wdo_prices),
                "baseline_depth": len(pair.corr_history),
                "latest_corr": round(pair.corr_history[-1], 3) if pair.corr_history else 0,
            }
        return out


# Singleton accessor
_cross_corr_instance: Optional[CrossCorrelationTracker] = None


def get_cross_correlation_tracker() -> CrossCorrelationTracker:
    """Get the singleton cross-correlation tracker."""
    global _cross_corr_instance
    if _cross_corr_instance is None:
        _cross_corr_instance = CrossCorrelationTracker()
    return _cross_corr_instance
