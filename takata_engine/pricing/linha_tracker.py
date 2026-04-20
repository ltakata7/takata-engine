"""Linha (cupom cambial) tracker — EM stress / on-off spread signal.

The cupom cambial ("Linha") is the implied offshore USD rate derived from
the onshore DI curve and the dollar-futures price (BNDF1 formula from
Lauro's institutional VBA spreadsheet):

    Linha = (onshore_factor * Spot / DolFut - 1) * 360 / dias_linha

Brought live into `DICurve.cupom_cambial()` — see
`takata_engine/pricing/di_curve.py:225`.

Why it matters for WDO signals:
- Rising Linha = foreign investors paying MORE for onshore BRL access
  → EM risk-off → BRL weakens → WDO rises
- Falling Linha = normalization → BRL strengthens → WDO falls
- Rate-of-change of Linha is the near-real-time proxy for on/off spread
  widening/tightening without needing an offshore NDF feed.

This tracker mirrors the `VXBRTracker` pattern from `implied_vol.py`:
- Module-level deque for rolling history (15k readings ≈ 8h at 2s polling)
- Daily reset at midnight
- `LinhaTracker.update(cupom)` — record a reading
- `LinhaTracker.signal(cupom)` — compute 5m/30m rate-of-change and fire reasons
- `get_linha_tracker()` singleton accessor

Fires these WDO signal reasons:
- ``linha_spike`` — roc_5m > +30 bps: bull (BRL weak, WDO up)
- ``linha_compression`` — roc_5m < -30 bps: bear
- ``on_off_widening`` — roc_30m > +50 bps: bull (sustained stress)
- ``on_off_tightening`` — roc_30m < -50 bps: bear (sustained normalization)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ── Module-level history (singleton state) ──────────────────────────
_LINHA_HISTORY_MAX = 15000  # ~8+ hours at 2s intervals
_linha_history: deque = deque(maxlen=_LINHA_HISTORY_MAX)
_linha_daily_reset: str = ""


# Signal thresholds (basis points)
_ROC_5M_SPIKE_BPS = 30.0
_ROC_30M_WIDENING_BPS = 50.0

# Score adjustments
_SCORE_ADJ_SPIKE = 0.10
_SCORE_ADJ_WIDENING = 0.08


class LinhaTracker:
    """Tracks cupom cambial (Linha) intraday for rate-of-change signals.

    Call ``update(cupom)`` every scan cycle. Call ``signal(cupom)`` for
    the analysis. Resets history daily. Persists intraday readings in a
    module-level deque.
    """

    def update(self, cupom: Optional[float], ts: Optional[float] = None) -> None:
        """Record a new cupom reading (percentage, e.g. 8.5 for 8.5%)."""
        global _linha_daily_reset
        if cupom is None:
            return
        try:
            cupom_f = float(cupom)
        except (TypeError, ValueError):
            return

        now = ts if ts is not None else time.time()
        today = time.strftime("%Y-%m-%d", time.localtime(now))

        # Daily reset
        if _linha_daily_reset != today:
            _linha_history.clear()
            _linha_daily_reset = today

        _linha_history.append((now, cupom_f))

    def _roc_bps(self, window_s: float, current: float) -> Optional[float]:
        """Rate of change over the given time window, in basis points.

        cupom values are in percentage form (e.g. 8.5), so the change in
        percentage points × 100 = change in bps.
        """
        if not _linha_history:
            return None
        target = time.time() - window_s
        # Find the oldest reading at or after `target` (left-to-right scan)
        past = None
        for t, c in _linha_history:
            if t >= target:
                past = (t, c)
                break
        if past is None:
            return None
        return (current - past[1]) * 100.0

    def signal(self, cupom: Optional[float]) -> Dict[str, Any]:
        """Full linha analysis: rate of change + signal reasons.

        Returns dict with:
        - ``cupom``: current reading
        - ``roc_5m``: rate of change in bps over last 5 minutes (or None)
        - ``roc_30m``: rate of change in bps over last 30 minutes (or None)
        - ``reasons``: list of linha-based signal reasons fired
        - ``score_adjustment``: total score delta from fired reasons
        """
        result: Dict[str, Any] = {
            "cupom": cupom,
            "roc_5m": None,
            "roc_30m": None,
            "reasons": [],
            "score_adjustment": 0.0,
        }

        if cupom is None:
            return result

        try:
            cupom_f = float(cupom)
        except (TypeError, ValueError):
            return result

        roc_5m = self._roc_bps(300, cupom_f)
        roc_30m = self._roc_bps(1800, cupom_f)
        result["roc_5m"] = None if roc_5m is None else round(roc_5m, 2)
        result["roc_30m"] = None if roc_30m is None else round(roc_30m, 2)

        reasons: list = result["reasons"]
        adj = 0.0

        if roc_5m is not None:
            if roc_5m > _ROC_5M_SPIKE_BPS:
                reasons.append("linha_spike")
                adj += _SCORE_ADJ_SPIKE
            elif roc_5m < -_ROC_5M_SPIKE_BPS:
                reasons.append("linha_compression")
                adj += _SCORE_ADJ_SPIKE

        if roc_30m is not None:
            if roc_30m > _ROC_30M_WIDENING_BPS:
                reasons.append("on_off_widening")
                adj += _SCORE_ADJ_WIDENING
            elif roc_30m < -_ROC_30M_WIDENING_BPS:
                reasons.append("on_off_tightening")
                adj += _SCORE_ADJ_WIDENING

        result["score_adjustment"] = adj
        return result


# Module-level singleton
_linha_tracker: Optional[LinhaTracker] = None


def get_linha_tracker() -> LinhaTracker:
    """Get or create the Linha tracker singleton."""
    global _linha_tracker
    if _linha_tracker is None:
        _linha_tracker = LinhaTracker()
    return _linha_tracker
