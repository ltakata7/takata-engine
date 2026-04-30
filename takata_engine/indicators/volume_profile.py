"""Volume Profile — Point of Control (POC), Value Area High (VAH), Value Area Low (VAL).

A volume profile bins traded volume by price across a window of bars and finds
the price levels where the most volume changed hands. Day-trade uses:

- **POC** acts as gravity. Price often returns to POC during the session.
- **VAH/VAL** mark the boundary of the price range that contains 70% of volume.
  Trades inside the value area are mean-reversion territory; trades that break
  out of it tend to continue in the breakout direction.
- **Prior-day POC/VAH/VAL** carry over as key reference levels for the next session.

Implementation notes:
- Each bar's volume is distributed across all bins between its high and low,
  proportionally to the bin's overlap with the bar range. This is more accurate
  than dropping all volume at typical price.
- Value Area is computed by expanding outward from POC (alternately above/below)
  until cumulative volume ≥ value_area_pct of total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class VolumeProfile:
    """Result of a volume profile computation.

    Attributes
    ----------
    poc : float
        Point of Control — price level with the highest traded volume.
    vah : float
        Value Area High — upper boundary of the value area.
    val : float
        Value Area Low — lower boundary of the value area.
    bin_edges : np.ndarray
        Price bin edges (length = len(bin_volumes) + 1).
    bin_volumes : np.ndarray
        Total volume in each bin.
    total_volume : float
        Sum of bin_volumes — should match input volume sum (modulo rounding).
    value_area_pct : float
        The percentage threshold used (default 0.7).
    """
    poc: float
    vah: float
    val: float
    bin_edges: np.ndarray
    bin_volumes: np.ndarray
    total_volume: float
    value_area_pct: float

    def to_dict(self) -> dict:
        """Lightweight dict for JSON / signal payloads."""
        return {
            "poc": float(self.poc),
            "vah": float(self.vah),
            "val": float(self.val),
            "total_volume": float(self.total_volume),
            "value_area_pct": float(self.value_area_pct),
            "n_bins": int(len(self.bin_volumes)),
        }


def volume_profile(
    df: pd.DataFrame,
    bins: int = 24,
    value_area_pct: float = 0.70,
    price_col_high: str = "high",
    price_col_low: str = "low",
    volume_col: str = "volume",
) -> VolumeProfile:
    """Compute volume profile (POC, VAH, VAL) over an OHLCV window.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV bars. Must contain `high`, `low`, `volume` columns.
    bins : int
        Number of price bins. 24 is a reasonable default for an intraday
        session; for prior-day reference levels, 30–50 is common.
    value_area_pct : float
        Fraction of total volume the value area must contain. 0.70 is the
        market-profile standard.
    price_col_high, price_col_low, volume_col : str
        Column names — override if your DataFrame uses different names.

    Returns
    -------
    VolumeProfile
        With POC, VAH, VAL, and the underlying histogram for charting.
    """
    if df.empty:
        return VolumeProfile(
            poc=0.0, vah=0.0, val=0.0,
            bin_edges=np.array([0.0]), bin_volumes=np.array([]),
            total_volume=0.0, value_area_pct=value_area_pct,
        )

    highs = df[price_col_high].to_numpy(dtype=float)
    lows = df[price_col_low].to_numpy(dtype=float)
    vols = df[volume_col].to_numpy(dtype=float)

    p_min = float(lows.min())
    p_max = float(highs.max())
    if p_max <= p_min:
        # Degenerate — single price. Drop everything in one bin.
        edges = np.array([p_min, p_max + 1e-9])
        return VolumeProfile(
            poc=p_min, vah=p_min, val=p_min,
            bin_edges=edges, bin_volumes=np.array([float(vols.sum())]),
            total_volume=float(vols.sum()), value_area_pct=value_area_pct,
        )

    edges = np.linspace(p_min, p_max, bins + 1)
    bin_volumes = np.zeros(bins, dtype=float)

    # Distribute each bar's volume across overlapping bins, proportional to
    # the fraction of the bar's range that falls in each bin. Vectorized
    # over bins; loops over bars (cheap — typically < 500 bars).
    bin_widths = np.diff(edges)
    for h, l, v in zip(highs, lows, vols):
        if v <= 0 or h <= l:
            # Zero volume or zero range — drop volume at midpoint.
            mid = (h + l) / 2
            idx = min(bins - 1, max(0, int((mid - p_min) / (p_max - p_min) * bins)))
            bin_volumes[idx] += v
            continue
        # Overlap of [l, h] with each bin [edges[i], edges[i+1]].
        overlap_lo = np.maximum(edges[:-1], l)
        overlap_hi = np.minimum(edges[1:], h)
        overlap = np.clip(overlap_hi - overlap_lo, 0, None)
        total_overlap = (h - l)  # always > 0 here
        if total_overlap > 0:
            bin_volumes += v * (overlap / total_overlap)

    total = float(bin_volumes.sum())
    if total <= 0:
        return VolumeProfile(
            poc=(p_min + p_max) / 2, vah=p_max, val=p_min,
            bin_edges=edges, bin_volumes=bin_volumes,
            total_volume=0.0, value_area_pct=value_area_pct,
        )

    # POC = bin with max volume; report price as bin center.
    poc_idx = int(np.argmax(bin_volumes))
    poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2

    # Value Area: expand outward from POC alternately, taking whichever side has
    # more volume next, until cumulative volume ≥ target.
    target = total * value_area_pct
    cum = bin_volumes[poc_idx]
    lo_idx = poc_idx
    hi_idx = poc_idx
    while cum < target and (lo_idx > 0 or hi_idx < bins - 1):
        # Volume of the candidate bins on each side (look 1 step out).
        below_vol = bin_volumes[lo_idx - 1] if lo_idx > 0 else -1.0
        above_vol = bin_volumes[hi_idx + 1] if hi_idx < bins - 1 else -1.0
        # Take whichever has more volume; ties go to the upside (matches the
        # market-profile convention used by most trading platforms).
        if above_vol >= below_vol and hi_idx < bins - 1:
            hi_idx += 1
            cum += bin_volumes[hi_idx]
        elif lo_idx > 0:
            lo_idx -= 1
            cum += bin_volumes[lo_idx]
        else:
            break  # both sides exhausted

    vah = float(edges[hi_idx + 1])
    val = float(edges[lo_idx])

    return VolumeProfile(
        poc=poc, vah=vah, val=val,
        bin_edges=edges, bin_volumes=bin_volumes,
        total_volume=total, value_area_pct=value_area_pct,
    )


def session_volume_profile(
    df: pd.DataFrame,
    session_col: Optional[str] = None,
    bins: int = 24,
    value_area_pct: float = 0.70,
) -> dict[object, VolumeProfile]:
    """Compute one volume profile per session.

    Returns a dict keyed by session label (date if `session_col` is None).
    Useful for prior-day POC: pass the full multi-day frame, take the last
    completed session's profile as the reference for today's open.
    """
    if df.empty:
        return {}
    if session_col and session_col in df.columns:
        groups = df.groupby(df[session_col], sort=False)
    else:
        groups = df.groupby(df.index.date, sort=False)
    return {
        key: volume_profile(g, bins=bins, value_area_pct=value_area_pct)
        for key, g in groups
    }
