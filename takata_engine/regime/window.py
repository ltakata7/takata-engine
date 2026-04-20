"""Sliding window utilities for converting time series into distributions."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def window_lift(
    series: np.ndarray | pd.Series,
    window_size: int = 30,
    stride: int = 5,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Slice a 1-D series into overlapping windows.

    Each window is treated as an empirical probability distribution
    for Wasserstein k-means clustering.

    Parameters
    ----------
    series : array-like
        1-D return series.
    window_size : int
        Number of observations per window.
    stride : int
        Step size between consecutive windows.

    Returns
    -------
    windows : list of np.ndarray
        Each element is a window of ``window_size`` observations.
    indices : list of (start, end) tuples
        Index ranges in the original series for each window.
    """
    arr = np.asarray(series, dtype=np.float64)
    n = len(arr)

    windows: List[np.ndarray] = []
    indices: List[Tuple[int, int]] = []

    start = 0
    while start + window_size <= n:
        end = start + window_size
        windows.append(arr[start:end])
        indices.append((start, end))
        start += stride

    return windows, indices


def reconstruct(
    returns: pd.Series,
    indices: List[Tuple[int, int]],
    assignments: np.ndarray | List[int],
) -> pd.DataFrame:
    """Map cluster assignments back to the original time index.

    For overlapping windows, each bar may belong to multiple windows.
    The assignment of the *last* window covering a bar is used (most
    recent regime classification wins).

    Parameters
    ----------
    returns : pd.Series
        Original return series with DatetimeIndex.
    indices : list of (start, end) tuples
        Window index ranges from :func:`window_lift`.
    assignments : array-like of int
        Cluster label for each window.

    Returns
    -------
    pd.DataFrame
        Columns: ``returns``, ``cumulative_returns``, ``regime``.
    """
    regime = np.full(len(returns), np.nan)

    for (start, end), label in zip(indices, assignments):
        regime[start:end] = label

    cumulative = (1 + returns).cumprod()

    return pd.DataFrame(
        {
            "returns": returns.values,
            "cumulative_returns": cumulative.values,
            "regime": regime,
        },
        index=returns.index,
    )
