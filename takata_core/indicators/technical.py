"""Core technical indicators — pure pandas/numpy implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int = 9) -> pd.Series:
    """Exponential Moving Average."""
    result = series.ewm(span=period, adjust=False).mean()
    result.name = f"ema_{period}"
    return result


def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    result = series.rolling(window=period).mean()
    result.name = f"sma_{period}"
    return result


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD — returns DataFrame with columns: macd, signal, histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }, index=series.index)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothed averages
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result.name = "rsi"
    return result


def bollinger(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands — returns DataFrame with columns: lower, middle, upper."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std

    return pd.DataFrame({
        "lower": lower,
        "middle": middle,
        "upper": upper,
    }, index=series.index)


def _true_range(df: pd.DataFrame) -> pd.Series:
    """True Range for ATR/ADX calculations."""
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    tr = _true_range(df)
    result = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    result.name = "atr"
    return result


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index — returns DataFrame with columns: adx, plus_di, minus_di."""
    high = df["high"]
    low = df["low"]

    # Directional movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = _true_range(df)

    # Wilder's smoothing
    alpha = 1 / period
    atr_smooth = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di = 100 * plus_dm_smooth / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr_smooth.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return pd.DataFrame({
        "adx": adx_line,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }, index=df.index)
