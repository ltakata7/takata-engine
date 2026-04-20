"""VWAP with session-resetting logic and standard deviation bands."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vwap(
    df: pd.DataFrame,
    session_col: str | None = None,
) -> pd.Series:
    """Session-resetting Volume-Weighted Average Price.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex.
    session_col : str, optional
        Column name containing session labels. If ``None``, resets daily
        (groups by date).

    Returns
    -------
    pd.Series
        VWAP values aligned with the input index.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical_price * df["volume"]

    if session_col and session_col in df.columns:
        groups = df[session_col]
    else:
        groups = df.index.date

    cum_tp_vol = tp_vol.groupby(groups).cumsum()
    cum_vol = df["volume"].groupby(groups).cumsum()

    result = cum_tp_vol / cum_vol.replace(0, np.nan)
    result.name = "vwap"
    return result


def vwap_bands(
    df: pd.DataFrame,
    num_std: list[float] | float = 2.0,
    session_col: str | None = None,
) -> pd.DataFrame:
    """VWAP with standard deviation bands.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex.
    num_std : float or list of float
        Number of standard deviations for bands. A single value creates
        ± that many std bands. A list creates multiple band pairs.
    session_col : str, optional
        Session grouping column (see :func:`vwap`).

    Returns
    -------
    pd.DataFrame
        Columns: ``vwap``, and for each std N: ``upper_N``, ``lower_N``.
    """
    if isinstance(num_std, (int, float)):
        num_std = [num_std]

    vwap_line = vwap(df, session_col)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0

    if session_col and session_col in df.columns:
        groups = df[session_col]
    else:
        groups = df.index.date

    # Expanding std of typical price within each session
    tp_std = typical_price.groupby(groups).expanding().std().droplevel(0)
    tp_std = tp_std.reindex(df.index)

    result = pd.DataFrame({"vwap": vwap_line}, index=df.index)
    for n in num_std:
        label = str(n).replace(".", "_")
        result[f"upper_{label}"] = vwap_line + n * tp_std
        result[f"lower_{label}"] = vwap_line - n * tp_std

    return result
