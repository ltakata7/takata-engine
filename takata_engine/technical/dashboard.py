"""Technical dashboard — Morgan Stanley institutional-style analysis note."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from takata_engine.data.feed_market import MarketDataFeed
from takata_engine.indicators.technical import adx, atr, bollinger, ema, macd, rsi, sma
from takata_engine.indicators.vwap import vwap
from takata_engine.technical.levels import (
    fibonacci_retracement,
    pivot_points,
    support_resistance,
)
from takata_engine.technical.patterns import PatternMatch, detect_patterns
from takata_engine.technical.trend import (
    classify_trend,
    multi_timeframe_summary,
    trend_label_short,
)

logger = logging.getLogger(__name__)


@dataclass
class DashboardResult:
    """Complete technical analysis result."""

    ticker: str
    date: str
    price: float
    change_pct: float

    # Trend
    trends: Dict[str, str]  # daily/weekly/monthly

    # Moving averages
    ma_values: Dict[str, float]  # ema_9, sma_20, sma_50, sma_200
    ma_crossovers: List[str]

    # Indicators
    rsi_value: float
    rsi_interpretation: str
    macd_value: float
    macd_signal_value: float
    macd_histogram: float
    macd_interpretation: str
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_interpretation: str
    adx_value: float
    atr_value: float

    # Volume
    volume_current: float
    volume_avg_20: float
    volume_interpretation: str

    # Levels
    support_levels: List[float]
    resistance_levels: List[float]
    pivot: Dict[str, float]
    fibonacci: Dict[str, float]

    # Patterns
    patterns: List[PatternMatch]

    # Trade setup
    trade_bias: str
    entry_zone: str
    stop_loss: float
    targets: List[float]


class TechnicalDashboard:
    """Generates a full technical analysis dashboard for any ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. ``"AAPL"``, ``"SPY"``).
    period : str
        Data period for daily data.
    """

    def __init__(self, ticker: str, period: str = "1y"):
        self.ticker = ticker.upper()
        self.period = period
        self._result: Optional[DashboardResult] = None

    def generate(self) -> DashboardResult:
        """Run full analysis and return structured result."""
        feed = MarketDataFeed(self.ticker, period=self.period, interval="1d")
        daily = feed.load()
        mtf_data = feed.load_multi_timeframe()

        close = daily["close"]
        price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = (price - prev_close) / prev_close * 100

        # ── Trend ──
        trends = multi_timeframe_summary(mtf_data)

        # ── Moving Averages ──
        ema_9 = float(ema(close, 9).iloc[-1])
        sma_20 = float(sma(close, 20).iloc[-1])
        sma_50 = float(sma(close, 50).iloc[-1])
        sma_200_val = float(sma(close, 200).iloc[-1]) if len(close) >= 200 else np.nan

        ma_values = {"EMA 9": ema_9, "SMA 20": sma_20, "SMA 50": sma_50, "SMA 200": sma_200_val}

        crossovers = []
        if ema_9 > sma_20 > sma_50:
            crossovers.append("Bullish EMA/SMA stack (9 > 20 > 50)")
        elif ema_9 < sma_20 < sma_50:
            crossovers.append("Bearish EMA/SMA stack (9 < 20 < 50)")
        if not np.isnan(sma_200_val):
            if price > sma_200_val:
                crossovers.append("Price ABOVE 200 SMA (long-term bullish)")
            else:
                crossovers.append("Price BELOW 200 SMA (long-term bearish)")
            if sma_50 > sma_200_val and len(close) >= 200:
                # Check if golden cross happened recently
                sma50_series = sma(close, 50)
                sma200_series = sma(close, 200)
                if len(sma50_series) > 5 and sma50_series.iloc[-5] < sma200_series.iloc[-5]:
                    crossovers.append("GOLDEN CROSS (50 SMA crossed above 200 SMA)")
            elif sma_50 < sma_200_val and len(close) >= 200:
                sma50_series = sma(close, 50)
                sma200_series = sma(close, 200)
                if len(sma50_series) > 5 and sma50_series.iloc[-5] > sma200_series.iloc[-5]:
                    crossovers.append("DEATH CROSS (50 SMA crossed below 200 SMA)")

        # ── RSI ──
        rsi_val = float(rsi(close, 14).iloc[-1])
        if rsi_val > 70:
            rsi_interp = "OVERBOUGHT — potential pullback"
        elif rsi_val > 60:
            rsi_interp = "Bullish momentum"
        elif rsi_val > 40:
            rsi_interp = "Neutral"
        elif rsi_val > 30:
            rsi_interp = "Bearish momentum"
        else:
            rsi_interp = "OVERSOLD — potential bounce"

        # ── MACD ──
        macd_df = macd(close, 12, 26, 9)
        macd_val = float(macd_df["macd"].iloc[-1])
        macd_sig = float(macd_df["signal"].iloc[-1])
        macd_hist = float(macd_df["histogram"].iloc[-1])
        prev_hist = float(macd_df["histogram"].iloc[-2])

        if macd_val > macd_sig and macd_hist > 0:
            if macd_hist > prev_hist:
                macd_interp = "Bullish — momentum ACCELERATING"
            else:
                macd_interp = "Bullish — momentum decelerating"
        elif macd_val < macd_sig and macd_hist < 0:
            if macd_hist < prev_hist:
                macd_interp = "Bearish — momentum ACCELERATING"
            else:
                macd_interp = "Bearish — momentum decelerating"
        else:
            macd_interp = "Crossover zone — watch for confirmation"

        # ── Bollinger Bands ──
        bb = bollinger(close, 20, 2.0)
        bb_u = float(bb["upper"].iloc[-1])
        bb_m = float(bb["middle"].iloc[-1])
        bb_l = float(bb["lower"].iloc[-1])
        bb_width = (bb_u - bb_l) / bb_m

        if price > bb_u:
            bb_interp = f"ABOVE upper band — extended, watch for mean reversion (width: {bb_width:.1%})"
        elif price < bb_l:
            bb_interp = f"BELOW lower band — oversold, watch for bounce (width: {bb_width:.1%})"
        elif bb_width < 0.04:
            bb_interp = f"Squeeze forming — expect volatility expansion (width: {bb_width:.1%})"
        else:
            position = (price - bb_l) / (bb_u - bb_l) * 100
            bb_interp = f"Within bands at {position:.0f}th percentile (width: {bb_width:.1%})"

        # ── ADX / ATR ──
        adx_df = adx(daily, 14)
        adx_val = float(adx_df["adx"].iloc[-1])
        atr_val = float(atr(daily, 14).iloc[-1])

        # ── Volume ──
        vol_current = float(daily["volume"].iloc[-1])
        vol_avg = float(daily["volume"].rolling(20).mean().iloc[-1])
        vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1.0

        if vol_ratio > 1.5:
            vol_interp = f"HIGH volume ({vol_ratio:.1f}x avg) — strong conviction"
        elif vol_ratio > 1.0:
            vol_interp = f"Above average ({vol_ratio:.1f}x avg)"
        elif vol_ratio > 0.7:
            vol_interp = f"Below average ({vol_ratio:.1f}x avg) — weak conviction"
        else:
            vol_interp = f"LOW volume ({vol_ratio:.1f}x avg) — caution"

        # ── Support / Resistance ──
        sr = support_resistance(daily, n_levels=3, order=10)
        pvt = pivot_points(daily)

        # ── Fibonacci ──
        high_52w = float(daily["high"].max())
        low_52w = float(daily["low"].min())
        fib = fibonacci_retracement(high_52w, low_52w)

        # ── Patterns ──
        patterns = detect_patterns(daily)

        # ── Trade Setup ──
        daily_trend = trends.get("daily", "sideways")
        weekly_trend = trends.get("weekly", "sideways")

        if "uptrend" in daily_trend and "uptrend" in weekly_trend:
            bias = "BULLISH"
            entry = f"Buy on pullback to {sma_20:.2f} (SMA 20) or {pvt['S1']:.2f} (S1)"
            stop = pvt["S2"]
            targets = [r for r in sr["resistance"][:2]] if sr["resistance"] else [price * 1.05]
        elif "downtrend" in daily_trend and "downtrend" in weekly_trend:
            bias = "BEARISH"
            entry = f"Short on rally to {sma_20:.2f} (SMA 20) or {pvt['R1']:.2f} (R1)"
            stop = pvt["R2"]
            targets = [s for s in sr["support"][:2]] if sr["support"] else [price * 0.95]
        else:
            bias = "NEUTRAL — wait for directional confirmation"
            entry = f"Range trade between {pvt['S1']:.2f} and {pvt['R1']:.2f}"
            stop = pvt["S2"]
            targets = [pvt["R1"]]

        self._result = DashboardResult(
            ticker=self.ticker,
            date=str(daily.index[-1].date()),
            price=price,
            change_pct=change_pct,
            trends=trends,
            ma_values=ma_values,
            ma_crossovers=crossovers,
            rsi_value=rsi_val,
            rsi_interpretation=rsi_interp,
            macd_value=macd_val,
            macd_signal_value=macd_sig,
            macd_histogram=macd_hist,
            macd_interpretation=macd_interp,
            bb_upper=bb_u,
            bb_middle=bb_m,
            bb_lower=bb_l,
            bb_interpretation=bb_interp,
            adx_value=adx_val,
            atr_value=atr_val,
            volume_current=vol_current,
            volume_avg_20=vol_avg,
            volume_interpretation=vol_interp,
            support_levels=sr["support"],
            resistance_levels=sr["resistance"],
            pivot=pvt,
            fibonacci=fib,
            patterns=patterns,
            trade_bias=bias,
            entry_zone=entry,
            stop_loss=stop,
            targets=targets,
        )
        return self._result

    def format_report(self) -> str:
        """Format the dashboard as a Morgan Stanley-style institutional note."""
        r = self._result
        if r is None:
            r = self.generate()

        w = 60
        lines = [
            "=" * w,
            f"  TECHNICAL ANALYSIS — {r.ticker}",
            f"  {r.date}  |  ${r.price:.2f}  ({r.change_pct:+.2f}%)",
            "=" * w,
            "",
            "TREND SUMMARY",
            "-" * w,
        ]
        for tf, trend in r.trends.items():
            lines.append(f"  {tf.capitalize():12s}  {trend_label_short(trend)}")

        lines.extend([
            "",
            "MOVING AVERAGES",
            "-" * w,
        ])
        for name, val in r.ma_values.items():
            if not np.isnan(val):
                pos = "ABOVE" if r.price > val else "BELOW"
                lines.append(f"  {name:10s}  ${val:>10.2f}  (price {pos})")
        for xover in r.ma_crossovers:
            lines.append(f"  >> {xover}")

        lines.extend([
            "",
            "MOMENTUM INDICATORS",
            "-" * w,
            f"  RSI (14):     {r.rsi_value:.1f}  — {r.rsi_interpretation}",
            f"  MACD:         {r.macd_value:.4f}  (signal: {r.macd_signal_value:.4f})",
            f"  MACD Hist:    {r.macd_histogram:+.4f}  — {r.macd_interpretation}",
            f"  ADX (14):     {r.adx_value:.1f}  {'(trending)' if r.adx_value > 25 else '(no trend)'}",
            "",
            "BOLLINGER BANDS (20, 2)",
            "-" * w,
            f"  Upper:  ${r.bb_upper:.2f}",
            f"  Middle: ${r.bb_middle:.2f}",
            f"  Lower:  ${r.bb_lower:.2f}",
            f"  >> {r.bb_interpretation}",
            "",
            "VOLUME",
            "-" * w,
            f"  Current:    {r.volume_current:,.0f}",
            f"  20-day avg: {r.volume_avg_20:,.0f}",
            f"  >> {r.volume_interpretation}",
            "",
            "SUPPORT & RESISTANCE",
            "-" * w,
        ])
        for i, level in enumerate(r.resistance_levels):
            lines.append(f"  R{i+1}:  ${level:.2f}")
        lines.append(f"  ---  ${r.price:.2f}  (current)")
        for i, level in enumerate(r.support_levels):
            lines.append(f"  S{i+1}:  ${level:.2f}")

        lines.extend([
            "",
            "PIVOT POINTS",
            "-" * w,
        ])
        for name in ["R3", "R2", "R1", "P", "S1", "S2", "S3"]:
            lines.append(f"  {name}:  ${r.pivot[name]:.2f}")

        lines.extend([
            "",
            "FIBONACCI RETRACEMENT (52-week range)",
            "-" * w,
        ])
        for name, val in r.fibonacci.items():
            marker = " <<" if abs(val - r.price) / r.price < 0.01 else ""
            lines.append(f"  {name:8s}  ${val:.2f}{marker}")

        if r.patterns:
            lines.extend([
                "",
                "PATTERN RECOGNITION",
                "-" * w,
            ])
            for p in r.patterns[:5]:
                lines.append(
                    f"  {p.name:20s}  {p.direction:8s}  conf={p.confidence:.0%}"
                )
                if p.description:
                    lines.append(f"    {p.description}")

        lines.extend([
            "",
            "TRADE SETUP",
            "=" * w,
            f"  Bias:     {r.trade_bias}",
            f"  Entry:    {r.entry_zone}",
            f"  Stop:     ${r.stop_loss:.2f}",
            f"  Targets:  {', '.join(f'${t:.2f}' for t in r.targets)}",
            "",
            f"  ATR (14): ${r.atr_value:.2f}  (daily volatility reference)",
            "=" * w,
        ])

        return "\n".join(lines)
