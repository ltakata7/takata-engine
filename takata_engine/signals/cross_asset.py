"""Cross-asset confluence — score signal direction against correlated instruments.

A WDO long (BRL weakening / USD strengthening) doesn't fire in a vacuum. If
ES is rallying hard, EM equities are bid, VIX is collapsing, and rates are
falling — the macro tape disagrees. The signal is statistically less likely
to work.

This module quantifies that. Given a primary instrument's direction and the
current state of its correlated peers, return a 0–100 confluence score plus
the per-peer breakdown so the trader can see *why*.

Correlation table is hand-curated from market structure (not statistically
derived). Magnitudes ∈ {1, 2} where 2 = "strong tell", 1 = "weak nudge".

Conventions:
- WDO ↑ means BRL weakens (USD/BRL futures rising).
- MES ↑ means US equities rising.
- DI1FUT is the front-month DI rate — DI ↑ = rates rising.
- **VIX is the primary global risk-off gauge for BOTH markets.** It's the
  CBOE Volatility Index on the S&P 500, deeply liquid, real-time, and the
  cleanest read of risk sentiment. EM (including BRL) tracks VIX more
  reliably than VXBR.
- VXBR (Brazil VIX) is kept as a weak secondary signal — it's illiquid,
  gappy, and often dominated by Petrobras options flow rather than broad
  Brazilian risk. Use it only to confirm VIX moves, never as the lead.
- DXY (when present) measures USD vs basket of majors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Direction-aware correlation weights. Sign convention:
#   +N → peer rising AGREES with primary direction (more confluence)
#   -N → peer rising DISAGREES (less confluence)
# Magnitude is the instrument's importance for that direction.
_CORRELATIONS: Dict[Tuple[str, str], Dict[str, int]] = {
    # WDO long = BRL weakening = capital leaving Brazil
    ("WDO", "long"): {
        "VIX":    +2,   # global vol spike = risk-off = capital out of EM = WDO up
        "ESFUT":  -2,   # global risk-off hurts BRL → ES falling agrees
        "NQFUT":  -2,   # tech leadership down = risk-off
        "INDFUT": -2,   # Brazil equities falling = capital flight = WDO up
        "WINFUT": -2,   # mini Bovespa, same as IND
        "IBOV":   -2,
        "VXBR":   +1,   # BR vol up — weak secondary (illiquid, often noisy)
        "DI1FUT": +1,   # DI rising can mean stress (carry vs flight ambiguous; weak weight)
        "BITFUT": -1,   # crypto risk-off proxy
    },
    # WDO short = BRL strengthening = inflows to Brazil
    ("WDO", "short"): {
        "VIX":    -2,   # vol crushing = risk-on = inflows to EM = WDO down
        "ESFUT":  +2,   # risk-on globally = capital into EM = BRL bid
        "NQFUT":  +2,
        "INDFUT": +2,   # Brazil equities up = inflows = WDO down
        "WINFUT": +2,
        "IBOV":   +2,
        "VXBR":   -1,   # weak secondary
        "DI1FUT": -1,   # DI falling = carry compresses, mild bear for BRL — small weight
        "BITFUT": +1,
    },
    # MES long = US equities rising
    ("MES", "long"): {
        "VIX":    -2,   # primary risk-off gauge — falling = risk-on = MES up
        "NQFUT":  +2,   # tech leads
        "ESFUT":  +2,   # full-size ES (basis arb)
        "INDFUT": +1,   # global equity beta
        "WINFUT": +1,
        "VXBR":   -1,   # weak proxy — only use to confirm VIX moves
        "BITFUT": +1,   # crypto risk-on, weak signal
    },
    # MES short = US equities falling
    ("MES", "short"): {
        "VIX":    +2,   # vol spiking = risk-off = MES down
        "NQFUT":  -2,
        "ESFUT":  -2,
        "INDFUT": -1,
        "WINFUT": -1,
        "VXBR":   +1,   # weak secondary
        "BITFUT": -1,
    },
}


@dataclass
class ConfluenceResult:
    """Result of a cross-asset confluence scoring."""

    score: int                      # 0–100, integer
    label: str                      # "strong" | "moderate" | "weak" | "disagree"
    agreed: List[Dict] = field(default_factory=list)
    disagreed: List[Dict] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)   # peers we wanted but lacked data
    raw_score: float = 0.0
    max_possible: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "label": self.label,
            "agreed": self.agreed,
            "disagreed": self.disagreed,
            "missing": self.missing,
            "raw_score": round(self.raw_score, 2),
            "max_possible": round(self.max_possible, 2),
        }


def _label_from_score(score: int) -> str:
    if score >= 70:
        return "strong"
    if score >= 50:
        return "moderate"
    if score >= 30:
        return "weak"
    return "disagree"


def cross_asset_confluence(
    instrument: str,
    direction: str,
    deltas_pct: Dict[str, float],
    threshold: float = 0.05,
) -> ConfluenceResult:
    """Score how much correlated instruments agree with the primary direction.

    Parameters
    ----------
    instrument : str
        Primary instrument: "WDO" or "MES".
    direction : str
        Primary signal direction: "long" or "short".
    deltas_pct : dict[str, float]
        Each peer's recent percent move (positive = peer rising). Keys are
        bridge tickers (e.g. "ESFUT", "VXBR", "DI1FUT", "INDFUT", "NQFUT").
        Pass `change_pct` from the bridge or any short-window momentum metric.
    threshold : float
        Minimum |delta_pct| to count a peer as "moved." Smaller moves are
        treated as flat (no agreement, no disagreement). 0.05 = 5 bps.

    Returns
    -------
    ConfluenceResult
        With score (0–100), label, per-peer breakdown, and missing peers.

    Examples
    --------
    >>> r = cross_asset_confluence("WDO", "long", {"ESFUT": -0.5, "VXBR": +1.2, "INDFUT": -0.3})
    >>> r.score >= 70
    True
    """
    key = (instrument.upper(), direction.lower())
    weights = _CORRELATIONS.get(key)
    if weights is None:
        return ConfluenceResult(score=50, label="moderate", raw_score=0.0, max_possible=0.0)

    raw = 0.0
    max_possible = 0.0
    agreed: List[Dict] = []
    disagreed: List[Dict] = []
    missing: List[str] = []

    for peer, weight in weights.items():
        max_possible += abs(weight)
        delta = deltas_pct.get(peer)
        if delta is None:
            missing.append(peer)
            continue
        if abs(delta) < threshold:
            # Flat: no contribution. Don't penalize, don't reward.
            continue
        # Sign of delta vs sign of weight. weight encodes "expected sign of
        # peer move when primary moves in `direction`". If signs match, it
        # agrees.
        agrees = (delta > 0) == (weight > 0)
        contribution = abs(weight) * (1 if agrees else -1)
        raw += contribution
        entry = {"peer": peer, "delta_pct": round(delta, 3), "weight": weight}
        (agreed if agrees else disagreed).append(entry)

    # Normalize raw ∈ [-max_possible, +max_possible] to score ∈ [0, 100].
    if max_possible == 0:
        score = 50
    else:
        score = int(round((raw / max_possible + 1) * 50))
        score = max(0, min(100, score))

    return ConfluenceResult(
        score=score,
        label=_label_from_score(score),
        agreed=agreed,
        disagreed=disagreed,
        missing=missing,
        raw_score=raw,
        max_possible=max_possible,
    )
