"""Macro Cycle Classifier — Prometheus-style regime detection.

Classifies the macro environment along 3 axes:
1. Growth: accelerating vs decelerating (equity trends, PMI proxies)
2. Inflation: rising vs falling (DI curve level, rate expectations)
3. Liquidity: easing vs tightening (DI curve slope, central bank stance)

These combine into 4 macro quadrants:
- Goldilocks: growth up, inflation down → risk-on, carry trades work
- Reflation: growth up, inflation up → commodities up, mixed FX
- Stagflation: growth down, inflation up → worst for EM, risk-off
- Deflation: growth down, inflation down → rate cuts, currency wars

Each quadrant implies directional biases for WDO, MES, DI, equities, etc.

Updated once per session (macro doesn't change intraday).
Signals from the cycle overlay add/subtract score from intraday signals.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CycleState:
    """Current macro cycle classification."""
    # Axes (-1 to +1 scale)
    growth: float = 0.0        # positive = accelerating, negative = decelerating
    inflation: float = 0.0     # positive = rising, negative = falling
    liquidity: float = 0.0     # positive = easing, negative = tightening

    # Classification
    regime: str = "unknown"    # goldilocks, reflation, stagflation, deflation, unknown
    confidence: float = 0.0    # 0-1 how clear the regime is

    # Instrument biases derived from regime (-1 to +1)
    wdo_bias: float = 0.0     # positive = WDO should rise (BRL weaken)
    mes_bias: float = 0.0     # positive = MES should rise (risk-on)
    di_bias: float = 0.0      # positive = DI should rise (rates up)
    equity_bias: float = 0.0  # positive = equities should rise

    # Input data used
    inputs: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "growth": round(self.growth, 2),
            "inflation": round(self.inflation, 2),
            "liquidity": round(self.liquidity, 2),
            "regime": self.regime,
            "confidence": round(self.confidence, 2),
            "biases": {
                "wdo": round(self.wdo_bias, 2),
                "mes": round(self.mes_bias, 2),
                "di": round(self.di_bias, 2),
                "equity": round(self.equity_bias, 2),
            },
            "inputs": self.inputs,
            "updated_at": self.updated_at,
        }


class MacroCycleClassifier:
    """Classifies the macro cycle from live market data.

    Uses data already available from the bridge:
    - WINFUT / INDFUT trend → growth proxy
    - ESFUT / MES trend → global growth proxy
    - DI curve (DI1FUT, DI1F27+) → inflation + liquidity
    - VXBR → risk/volatility
    - DOLFUT vs WDOFUT spread → flow pressure

    Call `update()` with current market data. Typically once at session start
    and again if major data releases occur.
    """

    def __init__(self):
        self.state = CycleState()
        self._last_update: float = 0

    def needs_update(self, interval: int = 300) -> bool:
        """Check if classification needs refreshing (default every 5 min)."""
        return time.time() - self._last_update > interval

    def update(
        self,
        # Equity trends (growth proxy)
        winfut_change_pct: float = 0,
        esfut_change_pct: float = 0,
        indfut_change_pct: float = 0,
        # DI curve (inflation + liquidity proxy)
        di_front_rate: float = 0,       # DI1FUT rate (e.g., 13.7%)
        di_front_change: float = 0,     # DI1FUT daily change %
        di_long_rate: float = 0,        # DI1F27/F28 rate
        di_slope_bps: float = 0,        # long - short in bps
        curve_shape: str = "unknown",   # from LiveDICurve
        # Volatility
        vxbr: float = 0,
        # FX
        dolfut_change_pct: float = 0,
        wdofut_change_pct: float = 0,
    ) -> CycleState:
        """Classify the macro cycle from current market data."""

        # ── Growth axis ──
        # Equity trends are the fastest growth proxy
        # WINFUT = Brazil growth, ESFUT = global growth
        growth_brazil = self._normalize(winfut_change_pct, scale=2.0)
        growth_global = self._normalize(esfut_change_pct, scale=1.5)
        growth = growth_brazil * 0.6 + growth_global * 0.4
        # Ibovespa index adds confirmation
        if indfut_change_pct != 0:
            growth = growth * 0.8 + self._normalize(indfut_change_pct, scale=2.0) * 0.2

        # ── Inflation axis ──
        # DI rate level = market's inflation expectation
        # DI change = direction of expectations
        # High DI + rising = inflation up. Low DI + falling = inflation down.
        if di_front_rate > 0:
            # Normalize DI rate: 10% = neutral, above = inflationary, below = deflationary
            rate_signal = (di_front_rate - 10.0) / 5.0  # 15% → +1, 5% → -1
            rate_signal = max(-1, min(1, rate_signal))

            # DI change direction
            change_signal = self._normalize(di_front_change, scale=0.5)

            inflation = rate_signal * 0.4 + change_signal * 0.6
        else:
            inflation = 0.0

        # ── Liquidity axis ──
        # Curve slope: steep = tightening expected, flat/inverted = easing expected
        # DI falling = easing, DI rising = tightening
        if di_slope_bps != 0:
            # Positive slope = tightening (rates expected higher), negative = easing
            slope_signal = self._normalize(di_slope_bps / 100, scale=1.0)
        else:
            slope_signal = 0.0

        # DI change as liquidity signal (falling = easing)
        liq_from_change = -self._normalize(di_front_change, scale=0.5)  # inverted: falling DI = easing

        # Curve shape adds conviction
        shape_signal = 0.0
        if curve_shape == "inverted" or curve_shape == "deeply_inverted":
            shape_signal = 0.5   # inverted = market expects easing
        elif curve_shape == "steep":
            shape_signal = -0.5  # steep = market expects tightening
        elif curve_shape == "flat":
            shape_signal = 0.2   # flat = near end of cycle, slight easing bias

        liquidity = liq_from_change * 0.4 + slope_signal * 0.3 + shape_signal * 0.3

        # VXBR adjusts: high vol = liquidity stress
        if vxbr > 25:
            liquidity -= 0.2  # high vol = tighter conditions
        elif vxbr > 0 and vxbr < 15:
            liquidity += 0.1  # low vol = ample liquidity

        # Clamp all axes to -1..+1
        growth = max(-1, min(1, growth))
        inflation = max(-1, min(1, inflation))
        liquidity = max(-1, min(1, liquidity))

        # ── Regime classification ──
        regime, confidence = self._classify_regime(growth, inflation, liquidity)

        # ── Instrument biases ──
        wdo_bias, mes_bias, di_bias, equity_bias = self._compute_biases(
            regime, growth, inflation, liquidity
        )

        self.state = CycleState(
            growth=growth,
            inflation=inflation,
            liquidity=liquidity,
            regime=regime,
            confidence=confidence,
            wdo_bias=wdo_bias,
            mes_bias=mes_bias,
            di_bias=di_bias,
            equity_bias=equity_bias,
            inputs={
                "winfut_chg": winfut_change_pct,
                "esfut_chg": esfut_change_pct,
                "di_front": di_front_rate,
                "di_front_chg": di_front_change,
                "di_long": di_long_rate,
                "di_slope_bps": di_slope_bps,
                "curve_shape": curve_shape,
                "vxbr": vxbr,
            },
            updated_at=time.time(),
        )
        self._last_update = time.time()

        logger.info(
            "Macro cycle: %s (conf=%.2f) | G=%.2f I=%.2f L=%.2f | WDO=%.2f MES=%.2f",
            regime, confidence, growth, inflation, liquidity, wdo_bias, mes_bias,
        )

        return self.state

    def update_from_indicators(
        self,
        # Market-derived (same as update())
        winfut_change_pct: float = 0,
        esfut_change_pct: float = 0,
        di_front_rate: float = 0,
        di_front_change: float = 0,
        di_slope_bps: float = 0,
        curve_shape: str = "unknown",
        vxbr: float = 0,
        # Real economic data (from BCB + FRED feeds)
        gdp_growth: float = 0,          # GDP y/y % (BCB 4380 or FRED GDPC1)
        cpi_change: float = 0,          # CPI m/m % (IPCA or CPIAUCSL)
        core_cpi_change: float = 0,     # Core CPI m/m %
        unemployment: float = 0,        # Unemployment rate %
        unemployment_change: float = 0, # Change in unemployment
        pmi: float = 0,                 # ISM/PMI index (>50 = expansion)
        fed_funds: float = 0,           # Fed Funds rate %
        selic: float = 0,              # Selic rate %
        us_10y2y_spread: float = 0,     # US 10Y-2Y spread
        inflation_expectation: float = 0, # Focus IPCA or breakeven
        rate_differential: float = 0,   # Selic - FedFunds
    ) -> CycleState:
        """Enhanced cycle classification using real economic data + market proxies.

        Blends actual GDP, CPI, unemployment with market-derived signals
        for higher-confidence regime classification.
        """
        # ── Growth axis: 60% market + 40% real data ──
        market_growth = (
            self._normalize(winfut_change_pct, scale=2.0) * 0.6 +
            self._normalize(esfut_change_pct, scale=1.5) * 0.4
        )

        real_growth = 0.0
        real_growth_sources = 0
        if gdp_growth != 0:
            # GDP: >2% = strong, 0-2% = moderate, <0 = recession
            real_growth += self._normalize(gdp_growth - 1.0, scale=3.0)
            real_growth_sources += 1
        if pmi > 0:
            # PMI: >50 expansion, <50 contraction
            real_growth += self._normalize(pmi - 50, scale=10.0)
            real_growth_sources += 1
        if unemployment_change != 0:
            # Rising unemployment = decelerating growth
            real_growth += -self._normalize(unemployment_change, scale=1.0)
            real_growth_sources += 1

        if real_growth_sources > 0:
            real_growth /= real_growth_sources
            growth = market_growth * 0.6 + real_growth * 0.4
        else:
            growth = market_growth

        # ── Inflation axis: 50% DI expectations + 50% actual readings ──
        market_inflation = 0.0
        if di_front_rate > 0:
            rate_signal = (di_front_rate - 10.0) / 5.0
            change_signal = self._normalize(di_front_change, scale=0.5)
            market_inflation = rate_signal * 0.4 + change_signal * 0.6

        real_inflation = 0.0
        real_inflation_sources = 0
        if cpi_change != 0:
            # CPI m/m: >0.3% = hot, 0.1-0.3% = normal, <0.1% = cool
            real_inflation += self._normalize(cpi_change - 0.2, scale=0.5)
            real_inflation_sources += 1
        if core_cpi_change != 0:
            real_inflation += self._normalize(core_cpi_change - 0.2, scale=0.4)
            real_inflation_sources += 1
        if inflation_expectation > 0:
            # Expectations above target (3% Brazil, 2% US) = inflationary
            target = 3.0 if selic > 5 else 2.0  # heuristic: high Selic = Brazil
            real_inflation += self._normalize(inflation_expectation - target, scale=2.0)
            real_inflation_sources += 1

        if real_inflation_sources > 0:
            real_inflation /= real_inflation_sources
            inflation = market_inflation * 0.5 + real_inflation * 0.5
        else:
            inflation = market_inflation

        # ── Liquidity axis: yield curve spreads + policy stance ──
        market_liquidity = 0.0
        if di_slope_bps != 0:
            market_liquidity += -self._normalize(di_slope_bps / 100, scale=1.0) * 0.3

        if di_front_change != 0:
            market_liquidity += -self._normalize(di_front_change, scale=0.5) * 0.3

        shape_signal = 0.0
        if curve_shape in ("inverted", "deeply_inverted"):
            shape_signal = 0.5
        elif curve_shape == "steep":
            shape_signal = -0.5
        market_liquidity += shape_signal * 0.2

        # Real data: yield curve spread as recession/easing signal
        if us_10y2y_spread != 0:
            # Negative spread = inversion = tightening that leads to easing
            spread_signal = self._normalize(us_10y2y_spread, scale=2.0)
            market_liquidity += spread_signal * 0.2

        if vxbr > 25:
            market_liquidity -= 0.2
        elif vxbr > 0 and vxbr < 15:
            market_liquidity += 0.1

        liquidity = max(-1, min(1, market_liquidity))

        # Clamp all
        growth = max(-1, min(1, growth))
        inflation = max(-1, min(1, inflation))

        # ── Classify ──
        regime, confidence = self._classify_regime(growth, inflation, liquidity)
        wdo_bias, mes_bias, di_bias, equity_bias = self._compute_biases(
            regime, growth, inflation, liquidity
        )

        # Track data quality
        data_sources = []
        if real_growth_sources > 0:
            data_sources.append("real_growth")
        if real_inflation_sources > 0:
            data_sources.append("real_inflation")
        if us_10y2y_spread != 0:
            data_sources.append("us_yield_curve")
        if rate_differential != 0:
            data_sources.append("rate_differential")
        data_sources.append("market_proxies")

        self.state = CycleState(
            growth=growth,
            inflation=inflation,
            liquidity=liquidity,
            regime=regime,
            confidence=confidence,
            wdo_bias=wdo_bias,
            mes_bias=mes_bias,
            di_bias=di_bias,
            equity_bias=equity_bias,
            inputs={
                "market": {"winfut_chg": winfut_change_pct, "esfut_chg": esfut_change_pct,
                           "di_front": di_front_rate, "di_front_chg": di_front_change,
                           "di_slope_bps": di_slope_bps, "vxbr": vxbr},
                "real": {"gdp": gdp_growth, "cpi": cpi_change, "core_cpi": core_cpi_change,
                         "unemployment": unemployment, "pmi": pmi,
                         "inflation_exp": inflation_expectation},
                "rates": {"fed_funds": fed_funds, "selic": selic,
                          "us_10y2y": us_10y2y_spread, "differential": rate_differential},
                "data_sources": data_sources,
            },
            updated_at=time.time(),
        )
        self._last_update = time.time()

        logger.info(
            "Macro cycle (enhanced): %s (conf=%.2f) | G=%.2f I=%.2f L=%.2f | sources=%s",
            regime, confidence, growth, inflation, liquidity, ",".join(data_sources),
        )

        return self.state

    def _normalize(self, value: float, scale: float = 1.0) -> float:
        """Normalize a value to roughly -1..+1 range."""
        return max(-1, min(1, value / scale))

    def _classify_regime(self, growth: float, inflation: float, liquidity: float):
        """Map growth + inflation into one of 4 quadrants."""
        # Primary classification from growth + inflation
        if growth > 0.1 and inflation < -0.1:
            regime = "goldilocks"
            confidence = min(abs(growth), abs(inflation))
        elif growth > 0.1 and inflation > 0.1:
            regime = "reflation"
            confidence = min(abs(growth), abs(inflation))
        elif growth < -0.1 and inflation > 0.1:
            regime = "stagflation"
            confidence = min(abs(growth), abs(inflation))
        elif growth < -0.1 and inflation < -0.1:
            regime = "deflation"
            confidence = min(abs(growth), abs(inflation))
        else:
            # Ambiguous — near zero on both axes
            regime = "transition"
            confidence = 0.2

        # Liquidity modifies confidence
        # In goldilocks, easing liquidity confirms. In stagflation, tight liquidity confirms.
        if regime == "goldilocks" and liquidity > 0:
            confidence *= 1.2
        elif regime == "stagflation" and liquidity < 0:
            confidence *= 1.2
        elif regime == "deflation" and liquidity > 0:
            confidence *= 1.1  # easing in deflation = policy response

        confidence = min(1.0, confidence)
        return regime, confidence

    def _compute_biases(self, regime: str, growth: float, inflation: float, liquidity: float):
        """Derive instrument biases from the macro regime.

        Returns (wdo_bias, mes_bias, di_bias, equity_bias).
        Positive = expect instrument to rise.
        """
        if regime == "goldilocks":
            # Best for BRL (carry + growth): WDO falls, equities rise, DI falls
            wdo = -0.3 - abs(growth) * 0.2   # BRL strengthens
            mes = 0.3 + abs(growth) * 0.2     # risk-on
            di = -0.2                          # rates can fall
            eq = 0.4 + abs(growth) * 0.2      # equities love goldilocks

        elif regime == "reflation":
            # Mixed: growth good but inflation rising = BCB may hike
            wdo = 0.0                          # neutral — opposing forces
            mes = 0.2                          # growth supports
            di = 0.3 + abs(inflation) * 0.2    # DI rises on inflation
            eq = 0.1                           # equities ok but rates headwind

        elif regime == "stagflation":
            # Worst for BRL and equities: WDO rises hard
            wdo = 0.4 + abs(inflation) * 0.2   # BRL weakens
            mes = -0.3 - abs(growth) * 0.2     # risk-off
            di = 0.2                            # rates stay high
            eq = -0.4 - abs(growth) * 0.2      # equities suffer

        elif regime == "deflation":
            # Rate cuts expected: BRL weakens but orderly
            wdo = 0.2                           # BRL weakens on rate cuts
            mes = -0.1                          # risk-off lite
            di = -0.4 - abs(inflation) * 0.2   # DI falls on cuts
            eq = -0.2                           # equities weak but recovering

        else:  # transition
            wdo = 0.0
            mes = 0.0
            di = 0.0
            eq = 0.0

        # Liquidity modifies: easing helps all risk assets
        if liquidity > 0.3:
            mes += 0.1
            eq += 0.1
            wdo -= 0.1  # easing supports BRL carry
        elif liquidity < -0.3:
            mes -= 0.1
            eq -= 0.1
            wdo += 0.1  # tightening hurts BRL

        return (
            max(-1, min(1, wdo)),
            max(-1, min(1, mes)),
            max(-1, min(1, di)),
            max(-1, min(1, eq)),
        )
