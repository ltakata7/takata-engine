"""Macro-regime-weighted position sizing.

Adjusts recommended position size based on the macro cycle state:
- In high-confidence regimes aligned with the signal direction → full size
- In transition/uncertain regimes → shrink size (variance control)
- In regimes contra the signal direction → heavy shrink (fighting macro)

This directly serves the consistency-over-upside trading goal: smaller sizes
when macro conviction is weak means smaller losses on bad days, and therefore
lower daily P&L variance. You sacrifice a small amount of upside on the
best days in exchange for meaningfully smaller drawdowns.

Returns a single multiplier in [0, 1] that the caller applies to the base
recommended size. Pure function, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MacroSizingConfig:
    """Knobs for how aggressive the macro sizing adjustment should be.

    All multipliers are in [0, 1]. Defaults are tuned for consistency —
    aggressive traders may want to raise the floors, conservative traders
    may want to lower them further.
    """
    # Applied when regime is "transition" or confidence is very low
    transition_multiplier: float = 0.50

    # Minimum multiplier when signal fights a high-confidence macro bias
    contra_floor: float = 0.40

    # Above this confidence, alignment stops earning bonus
    high_confidence_threshold: float = 0.70

    # Minimum multiplier in any "known" regime (floor for the whole function)
    absolute_floor: float = 0.30

    # Instrument bias thresholds for classifying alignment
    aligned_threshold: float = 0.30    # above this = clearly aligned
    contra_threshold: float = -0.30    # below this = clearly contra


DEFAULT_CONFIG = MacroSizingConfig()


def compute_macro_sizing_multiplier(
    regime: str,
    confidence: float,
    instrument_bias: float,
    signal_direction: str,
    config: MacroSizingConfig = DEFAULT_CONFIG,
) -> Dict[str, object]:
    """Compute a position-size multiplier from the current macro cycle state.

    Parameters
    ----------
    regime : str
        One of: "goldilocks", "reflation", "stagflation", "deflation",
        "transition", "not_initialized", "error". Unknown values treated as neutral.
    confidence : float
        Cycle classifier confidence [0, 1]. Higher = stronger conviction.
    instrument_bias : float
        The instrument's directional bias from the cycle state [-1, +1].
        Positive = macro expects instrument to rise. Negative = macro expects fall.
        For a long signal, positive bias = aligned. For a short signal, negative bias = aligned.
    signal_direction : str
        "long" or "short".
    config : MacroSizingConfig
        Sizing config. Use DEFAULT_CONFIG unless you have specific knobs.

    Returns
    -------
    dict with keys:
        "multiplier" (float) : the size multiplier in [absolute_floor, 1.0]
        "reason" (str)       : short label explaining which branch fired
        "alignment" (str)    : "aligned", "neutral", "contra"
        "regime" (str)       : pass-through of input regime
        "confidence" (float) : pass-through of input confidence
    """
    # Fall-through: no macro data → full size (don't penalize when we know nothing)
    if regime in ("not_initialized", "error", "unknown"):
        return {
            "multiplier": 1.0,
            "reason": "no_macro_data",
            "alignment": "unknown",
            "regime": regime,
            "confidence": confidence,
        }

    # Compute directional alignment: does the signal fight or follow the macro bias?
    # Long signal wants positive bias; short signal wants negative bias.
    if signal_direction == "long":
        directional_score = instrument_bias
    elif signal_direction == "short":
        directional_score = -instrument_bias
    else:
        # "exit" signals or unknown directions → full size (not a sizing decision)
        return {
            "multiplier": 1.0,
            "reason": "non_entry_signal",
            "alignment": "n/a",
            "regime": regime,
            "confidence": confidence,
        }

    # Classify alignment
    if directional_score >= config.aligned_threshold:
        alignment = "aligned"
    elif directional_score <= config.contra_threshold:
        alignment = "contra"
    else:
        alignment = "neutral"

    # Transition regime: variance is high by definition → shrink regardless of alignment
    if regime == "transition":
        mult = config.transition_multiplier
        if alignment == "aligned":
            # Small bonus for alignment even in transition
            mult = min(1.0, mult + 0.10)
        return {
            "multiplier": round(mult, 2),
            "reason": "transition_regime",
            "alignment": alignment,
            "regime": regime,
            "confidence": confidence,
        }

    # Known regime (goldilocks / reflation / stagflation / deflation)
    # Base multiplier scales with confidence: 0.6 at conf=0, 1.0 at conf=1
    base = 0.60 + 0.40 * min(1.0, confidence / config.high_confidence_threshold)

    if alignment == "aligned":
        mult = base
    elif alignment == "neutral":
        # Mild penalty for neutral alignment in a known regime
        mult = base * 0.85
    else:  # contra
        # Heavy penalty for fighting macro — scales with confidence
        # High-confidence contra = near the floor. Low-confidence contra = softer.
        penalty = 1.0 - (confidence * 0.6)
        mult = max(config.contra_floor, base * penalty)

    # Enforce absolute floor
    mult = max(config.absolute_floor, min(1.0, mult))

    return {
        "multiplier": round(mult, 2),
        "reason": f"{regime}_{alignment}",
        "alignment": alignment,
        "regime": regime,
        "confidence": confidence,
    }


def compute_recommended_contracts(
    base_contracts: float,
    multiplier_result: Dict[str, object],
) -> int:
    """Apply the multiplier to a base contract count, rounded down.

    Rounded DOWN (not nearest) because when in doubt, smaller size is safer
    for the consistency goal. Minimum is 1 contract if base >= 1.
    """
    mult = float(multiplier_result.get("multiplier", 1.0))
    raw = base_contracts * mult
    contracts = int(raw)  # truncate
    if base_contracts >= 1 and contracts < 1:
        contracts = 1
    return contracts
