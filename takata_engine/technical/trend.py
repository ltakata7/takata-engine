"""Multi-timeframe trend classification."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from takata_engine.indicators.technical import adx, ema


def classify_trend(df: pd.DataFrame, short: int = 20, medium: int = 50, long: int = 200) -> str:
    """Classify trend based on EMA structure and price position.

    Returns
    -------
    str
        ``"strong_uptrend"``, ``"uptrend"``, ``"sideways"``,
        ``"downtrend"``, or ``"strong_downtrend"``.
    """
    close = df["close"]
    if len(close) < long:
        # Not enough data for full analysis, use what we have
        if len(close) < short:
            return "sideways"
        ema_s = ema(close, min(short, len(close) - 1))
        return "uptrend" if close.iloc[-1] > ema_s.iloc[-1] else "downtrend"

    ema_short = ema(close, short).iloc[-1]
    ema_med = ema(close, medium).iloc[-1]
    ema_long = ema(close, long).iloc[-1]
    price = close.iloc[-1]

    # EMA stacking
    bullish_stack = ema_short > ema_med > ema_long
    bearish_stack = ema_short < ema_med < ema_long

    # ADX for trend strength
    adx_df = adx(df, 14)
    adx_val = adx_df["adx"].iloc[-1]
    strong = not np.isnan(adx_val) and adx_val > 25

    if bullish_stack and price > ema_short:
        return "strong_uptrend" if strong else "uptrend"
    elif bearish_stack and price < ema_short:
        return "strong_downtrend" if strong else "downtrend"
    else:
        return "sideways"


def trend_label_short(trend: str) -> str:
    """Short display label for trend."""
    return {
        "strong_uptrend": "STRONG UP",
        "uptrend": "UP",
        "sideways": "SIDEWAYS",
        "downtrend": "DOWN",
        "strong_downtrend": "STRONG DOWN",
    }.get(trend, trend.upper())


def multi_timeframe_summary(data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """Classify trend for each timeframe.

    Parameters
    ----------
    data : dict
        Keys: ``"daily"``, ``"weekly"``, ``"monthly"`` → OHLCV DataFrames.

    Returns
    -------
    dict
        Timeframe → trend label.
    """
    result = {}
    for tf, df in data.items():
        if df is not None and len(df) > 20:
            result[tf] = classify_trend(df)
        else:
            result[tf] = "insufficient_data"
    return result
