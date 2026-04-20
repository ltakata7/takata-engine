"""Yield Curve Analytics — combined US Treasury + Brazil DI curve analysis.

Computes spread regimes, rate differentials, and recession signals
from both US and Brazilian yield curves.

Key outputs for trading:
- US 10Y-2Y and 10Y-3M inversion = recession signal → risk-off → WDO up
- Rate differential (Selic - FedFunds) = carry trade driver → BRL strength/weakness
- Curve regime: normal / us_inversion / br_tightening / global_easing / divergent
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from . import EconomicIndicator, IndicatorBundle, MacroDataFeed

logger = logging.getLogger(__name__)


class YieldCurveAnalytics(MacroDataFeed):
    """Combines US Treasury and Brazil DI curve data."""

    source = "yield_curve"

    def analyze(
        self,
        fred_bundle: Optional[IndicatorBundle] = None,
        bcb_bundle: Optional[IndicatorBundle] = None,
        di_curve_data: Optional[Dict] = None,
    ) -> IndicatorBundle:
        """Analyze yield curves from FRED + BCB + bridge DI curve data.

        Parameters
        ----------
        fred_bundle : IndicatorBundle
            FRED data with DGS2, DGS10, DGS30, T10Y2Y, T10Y3M, FEDFUNDS
        bcb_bundle : IndicatorBundle
            BCB data with Selic (432)
        di_curve_data : dict
            From LiveDICurve.to_dict() — DI curve shape and tenors
        """
        bundle = IndicatorBundle(source="yield_curve")

        # ── US Treasury Curve ──
        us_2y = self._get_value(fred_bundle, "DGS2")
        us_10y = self._get_value(fred_bundle, "DGS10")
        us_30y = self._get_value(fred_bundle, "DGS30")
        us_spread_10y2y = self._get_value(fred_bundle, "T10Y2Y")
        us_spread_10y3m = self._get_value(fred_bundle, "T10Y3M")
        fed_funds = self._get_value(fred_bundle, "FEDFUNDS")

        # If we have FRED data, compute US curve metrics
        if us_10y and us_2y:
            spread = us_spread_10y2y if us_spread_10y2y else us_10y - us_2y

            # US curve inversion signal
            if spread < -0.2:
                us_signal = "bearish"
                us_interp = f"US curve INVERTED ({spread:+.2f}%) — recession signal"
            elif spread < 0:
                us_signal = "bearish"
                us_interp = f"US curve near inversion ({spread:+.2f}%) — late cycle"
            elif spread < 0.5:
                us_signal = "neutral"
                us_interp = f"US curve flat ({spread:+.2f}%) — uncertain"
            else:
                us_signal = "bullish"
                us_interp = f"US curve normal ({spread:+.2f}%) — healthy"

            bundle.indicators.append(EconomicIndicator(
                series_id="US_10Y2Y_SPREAD",
                name="US 10Y-2Y Spread",
                source="yield_curve", category="rates",
                latest_value=spread, unit="%",
                signal=us_signal, interpretation=us_interp,
            ))

        if us_spread_10y3m is not None:
            sig = "bearish" if us_spread_10y3m < 0 else "bullish"
            bundle.indicators.append(EconomicIndicator(
                series_id="US_10Y3M_SPREAD",
                name="US 10Y-3M Spread",
                source="yield_curve", category="rates",
                latest_value=us_spread_10y3m, unit="%",
                signal=sig,
                interpretation=f"{'Inverted' if us_spread_10y3m < 0 else 'Normal'} — strongest recession predictor",
            ))

        # ── Brazil DI Curve ──
        selic = self._get_value(bcb_bundle, "BCB_432")
        di_shape = "unknown"
        di_slope_bps = 0

        if di_curve_data and di_curve_data.get("loaded"):
            shape_info = di_curve_data.get("shape", {})
            di_shape = shape_info.get("shape", "unknown")
            di_slope_bps = shape_info.get("slope_bps", 0)
            di_short = shape_info.get("short_rate", 0)
            di_long = shape_info.get("long_rate", 0)

            if di_shape in ("inverted", "deeply_inverted"):
                br_signal = "bullish"
                br_interp = f"DI curve inverted (slope {di_slope_bps:+.0f}bps) — market expects rate cuts"
            elif di_shape == "steep":
                br_signal = "bearish"
                br_interp = f"DI curve steep (slope {di_slope_bps:+.0f}bps) — market expects rate hikes"
            elif di_shape == "flat":
                br_signal = "neutral"
                br_interp = f"DI curve flat — end of cycle signals"
            else:
                br_signal = "neutral"
                br_interp = f"DI curve normal (slope {di_slope_bps:+.0f}bps)"

            bundle.indicators.append(EconomicIndicator(
                series_id="BR_DI_SLOPE",
                name="Brazil DI Curve Slope",
                source="yield_curve", category="rates",
                latest_value=di_slope_bps / 100, unit="%",
                signal=br_signal, interpretation=br_interp,
            ))

            # Add DI curve tenors
            for pt in di_curve_data.get("points", []):
                bundle.indicators.append(EconomicIndicator(
                    series_id=f"DI_{pt['ticker']}",
                    name=f"DI {pt['ticker']}",
                    source="yield_curve", category="rates",
                    latest_value=pt["rate"], unit="%",
                    frequency="daily",
                ))

        # ── Rate Differential (Carry Trade) ──
        if selic and fed_funds:
            diff = selic - fed_funds
            carry_signal = "bullish" if diff > 5 else "neutral" if diff > 2 else "bearish"
            carry_interp = (
                f"Selic ({selic:.2f}%) - Fed Funds ({fed_funds:.2f}%) = {diff:+.2f}% — "
                f"{'attractive' if diff > 5 else 'moderate' if diff > 2 else 'low'} carry for BRL"
            )

            bundle.indicators.append(EconomicIndicator(
                series_id="RATE_DIFFERENTIAL",
                name="Selic - Fed Funds Differential",
                source="yield_curve", category="rates",
                latest_value=diff, unit="%",
                signal=carry_signal, interpretation=carry_interp,
            ))

        # ── Spread Regime Classification ──
        regime = self._classify_spread_regime(
            us_spread=us_spread_10y2y,
            di_slope_bps=di_slope_bps,
            di_shape=di_shape,
        )
        bundle.indicators.append(EconomicIndicator(
            series_id="SPREAD_REGIME",
            name="Global Spread Regime",
            source="yield_curve", category="rates",
            latest_value=0, unit="regime",
            signal=regime["signal"],
            interpretation=regime["interpretation"],
        ))

        return bundle

    def _get_value(self, bundle: Optional[IndicatorBundle], series_id: str) -> Optional[float]:
        """Extract a value from an IndicatorBundle."""
        if not bundle:
            return None
        ind = bundle.get(series_id)
        return ind.latest_value if ind else None

    def _classify_spread_regime(
        self,
        us_spread: Optional[float],
        di_slope_bps: float,
        di_shape: str,
    ) -> Dict[str, str]:
        """Classify the combined US + Brazil yield curve regime."""
        us_inverted = us_spread is not None and us_spread < 0
        br_tightening = di_shape == "steep" or di_slope_bps > 100
        br_easing = di_shape in ("inverted", "deeply_inverted")

        if us_inverted and br_tightening:
            return {
                "regime": "divergent",
                "signal": "bearish",
                "interpretation": "US recession signal + Brazil tightening — worst for EM risk",
            }
        elif us_inverted and br_easing:
            return {
                "regime": "global_easing",
                "signal": "neutral",
                "interpretation": "Both curves signaling rate cuts — coordinated easing cycle",
            }
        elif us_inverted:
            return {
                "regime": "us_inversion",
                "signal": "bearish",
                "interpretation": "US curve inverted — recession risk, risk-off for EM",
            }
        elif br_tightening:
            return {
                "regime": "br_tightening",
                "signal": "bearish",
                "interpretation": "Brazil DI curve steep — BCB expected to hike, BRL mixed",
            }
        else:
            return {
                "regime": "normal",
                "signal": "bullish",
                "interpretation": "Both curves normal — healthy global conditions",
            }
