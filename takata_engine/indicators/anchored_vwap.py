"""Anchored VWAP — VWAP starting from an arbitrary anchor point.

Unlike session VWAP (resets daily), anchored VWAP runs from a chosen event:
the open, prior-day high/low, a Fed/COPOM decision spike, the start of a
trend leg, etc. Each anchor produces a magnetic level that traders watch.

Common anchors and what they tell you:
- **Session open**: same as session VWAP, but explicit.
- **Prior-day high/low**: institutional reference. Touches often reverse.
- **News spike (FOMC, CPI, payrolls)**: anchors that level as a memory of
  the move. Reclaims/rejections of this AVWAP are high-probability signals.
- **Swing low (uptrend) / swing high (downtrend)**: tracks average entry
  price of participants who joined the trend at that pivot.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


def anchored_vwap(
    df: pd.DataFrame,
    anchor: Union[int, pd.Timestamp, str],
) -> pd.Series:
    """Volume-weighted average price starting from `anchor` and running forward.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with `high`, `low`, `close`, `volume`. Index can be anything.
    anchor : int | pd.Timestamp | str
        - int: positional index (0-based row offset).
        - Timestamp / str: looked up in `df.index`. If exact match isn't
          present, uses the first row at or after the anchor time.

    Returns
    -------
    pd.Series
        Anchored VWAP aligned with df.index. Values before the anchor are NaN.
        After the anchor, each value is the running VWAP from anchor through
        that row.
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index, name="anchored_vwap")

    if isinstance(anchor, int):
        anchor_pos = anchor
    else:
        if isinstance(anchor, str):
            anchor = pd.Timestamp(anchor)
        # Find first index >= anchor.
        if isinstance(df.index, pd.DatetimeIndex):
            mask = df.index >= anchor
            if not mask.any():
                # Anchor is past end of data — return all NaN.
                return pd.Series(np.nan, index=df.index, name="anchored_vwap")
            anchor_pos = int(np.argmax(mask))
        else:
            # Non-datetime index: try direct lookup, fall back to position 0.
            try:
                anchor_pos = df.index.get_loc(anchor)
            except KeyError:
                return pd.Series(np.nan, index=df.index, name="anchored_vwap")

    if anchor_pos < 0 or anchor_pos >= len(df):
        return pd.Series(np.nan, index=df.index, name="anchored_vwap")

    sub = df.iloc[anchor_pos:]
    typical = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    tp_vol = (typical * sub["volume"]).cumsum()
    cum_vol = sub["volume"].cumsum().replace(0, np.nan)
    avwap = tp_vol / cum_vol

    result = pd.Series(np.nan, index=df.index, name="anchored_vwap")
    result.iloc[anchor_pos:] = avwap.to_numpy()
    return result


def anchored_vwap_from_session_open(df: pd.DataFrame) -> pd.Series:
    """Convenience: AVWAP anchored at the first bar of the most recent session.

    Looks at the last date in the index and anchors at that session's first row.
    Useful when you want today's intraday AVWAP without managing the anchor
    yourself.
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(np.nan, index=df.index, name="anchored_vwap_session")
    last_date = df.index[-1].date()
    mask = df.index.date == last_date
    if not mask.any():
        return pd.Series(np.nan, index=df.index, name="anchored_vwap_session")
    anchor_pos = int(np.argmax(mask))
    s = anchored_vwap(df, anchor_pos)
    s.name = "anchored_vwap_session"
    return s


def anchored_vwap_from_prior_day_extreme(
    df: pd.DataFrame,
    extreme: str = "high",
) -> pd.Series:
    """AVWAP anchored at the prior session's high (or low) bar.

    Parameters
    ----------
    extreme : "high" | "low"
        Which extreme of the prior session to anchor at.

    Returns
    -------
    pd.Series
        Anchored VWAP. NaN before the anchor bar.
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(np.nan, index=df.index, name=f"avwap_pd_{extreme}")

    dates = df.index.date
    unique_dates = sorted(set(dates))
    if len(unique_dates) < 2:
        return pd.Series(np.nan, index=df.index, name=f"avwap_pd_{extreme}")

    prior_date = unique_dates[-2]
    prior_mask = dates == prior_date
    prior_slice = df.loc[prior_mask]

    if extreme == "high":
        anchor_label = prior_slice["high"].idxmax()
    elif extreme == "low":
        anchor_label = prior_slice["low"].idxmin()
    else:
        raise ValueError(f"extreme must be 'high' or 'low', got {extreme!r}")

    anchor_pos = df.index.get_loc(anchor_label)
    if isinstance(anchor_pos, slice):
        anchor_pos = anchor_pos.start
    s = anchored_vwap(df, int(anchor_pos))
    s.name = f"avwap_pd_{extreme}"
    return s
