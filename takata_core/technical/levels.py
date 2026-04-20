"""Support/resistance levels and Fibonacci analysis."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
    """Compute Fibonacci retracement levels between a swing high and low.

    Parameters
    ----------
    high : float
        Swing high price.
    low : float
        Swing low price.

    Returns
    -------
    dict
        Fibonacci level name → price.
    """
    diff = high - low
    return {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.500 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100.0%": low,
    }


def fibonacci_extension(
    high: float, low: float, retracement_low: float
) -> Dict[str, float]:
    """Compute Fibonacci extension levels.

    Parameters
    ----------
    high : float
        Initial swing high.
    low : float
        Initial swing low.
    retracement_low : float
        The retracement point from which extensions project.
    """
    diff = high - low
    return {
        "100.0%": retracement_low + diff,
        "127.2%": retracement_low + 1.272 * diff,
        "161.8%": retracement_low + 1.618 * diff,
        "200.0%": retracement_low + 2.0 * diff,
        "261.8%": retracement_low + 2.618 * diff,
    }


def _find_swing_points(
    series: pd.Series, order: int = 5
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Detect swing highs and lows.

    Parameters
    ----------
    series : pd.Series
        Price series (typically close or high/low).
    order : int
        Number of bars on each side to confirm a swing point.

    Returns
    -------
    swing_highs, swing_lows : list of (index, price) tuples
    """
    highs = []
    lows = []
    arr = series.values

    for i in range(order, len(arr) - order):
        if all(arr[i] >= arr[i - j] for j in range(1, order + 1)) and \
           all(arr[i] >= arr[i + j] for j in range(1, order + 1)):
            highs.append((i, float(arr[i])))

        if all(arr[i] <= arr[i - j] for j in range(1, order + 1)) and \
           all(arr[i] <= arr[i + j] for j in range(1, order + 1)):
            lows.append((i, float(arr[i])))

    return highs, lows


def support_resistance(
    df: pd.DataFrame, n_levels: int = 5, order: int = 10
) -> Dict[str, List[float]]:
    """Detect support and resistance levels from swing points.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    n_levels : int
        Max number of levels to return for each side.
    order : int
        Swing point detection window.

    Returns
    -------
    dict
        ``"support"`` → list of prices, ``"resistance"`` → list of prices.
    """
    highs, lows = _find_swing_points(df["high"], order)
    _, lows_from_low = _find_swing_points(df["low"], order)

    current_price = df["close"].iloc[-1]

    # Resistance: swing highs above current price
    resistance = sorted(
        set(round(p, 2) for _, p in highs if p > current_price)
    )[:n_levels]

    # Support: swing lows below current price
    support = sorted(
        set(round(p, 2) for _, p in lows_from_low if p < current_price),
        reverse=True,
    )[:n_levels]

    return {"support": support, "resistance": resistance}


def pivot_points(df: pd.DataFrame) -> Dict[str, float]:
    """Classic pivot points from the last completed bar.

    Returns
    -------
    dict
        Keys: ``P``, ``R1``, ``R2``, ``R3``, ``S1``, ``S2``, ``S3``.
    """
    h = df["high"].iloc[-2]
    l = df["low"].iloc[-2]
    c = df["close"].iloc[-2]

    pivot = (h + l + c) / 3
    return {
        "P": round(pivot, 2),
        "R1": round(2 * pivot - l, 2),
        "R2": round(pivot + (h - l), 2),
        "R3": round(h + 2 * (pivot - l), 2),
        "S1": round(2 * pivot - h, 2),
        "S2": round(pivot - (h - l), 2),
        "S3": round(l - 2 * (h - pivot), 2),
    }
