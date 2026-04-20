"""Real-time regime detection using Wasserstein k-means."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from takata_engine.regime.mmd_scorer import cluster_quality
from takata_engine.regime.wasserstein_kmeans import WassersteinKMeans
from takata_engine.regime.window import reconstruct, window_lift

logger = logging.getLogger(__name__)

# Default regime label ordering: sorted by centroid volatility ascending
DEFAULT_LABELS = ["low_vol_trending", "high_vol_choppy", "crisis"]


class RegimeDetector:
    """Detects market regimes from OHLCV bar data.

    Computes log returns, slices them into rolling windows, fits
    Wasserstein k-means, and maps cluster indices to human-readable
    regime labels ordered by volatility.

    Parameters
    ----------
    k : int
        Number of regimes.
    window_size : int
        Number of bars per window (each window = one distribution).
    stride : int
        Bars between consecutive windows.
    labels : list of str, optional
        Regime names in order of increasing volatility.
        Default: ``["low_vol_trending", "high_vol_choppy", "crisis"]``.
    max_iter : int
        Max k-means iterations.
    tol : float
        Convergence tolerance.
    gamma : float
        RBF bandwidth for MMD.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        k: int = 3,
        window_size: int = 30,
        stride: int = 5,
        labels: Optional[List[str]] = None,
        max_iter: int = 50,
        tol: float = 1e-4,
        gamma: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.k = k
        self.window_size = window_size
        self.stride = stride
        self.labels = labels or DEFAULT_LABELS[:k]
        self.gamma = gamma
        self.random_state = random_state

        if len(self.labels) != k:
            raise ValueError(f"Expected {k} labels, got {len(self.labels)}")

        self.model = WassersteinKMeans(
            k=k,
            max_iter=max_iter,
            tol=tol,
            gamma=gamma,
            random_state=random_state,
        )
        self._label_map: Optional[Dict[int, str]] = None
        self._fitted = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RegimeDetector":
        """Construct from a regime config dict (e.g. ``cfg["regime"]``)."""
        return cls(
            k=config.get("k", 3),
            window_size=config.get("window_size", 30),
            stride=config.get("stride", 5),
            max_iter=config.get("max_iter", 50),
            tol=config.get("tol", 1e-4),
            gamma=config.get("gamma", 1.0),
            random_state=config.get("random_state", None),
        )

    @staticmethod
    def _log_returns(df: pd.DataFrame) -> pd.Series:
        """Compute log returns from close prices."""
        returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        return returns

    def _build_label_map(self) -> Dict[int, str]:
        """Map cluster indices to regime labels by centroid volatility."""
        vols = self.model.centroid_volatilities()
        # Sort cluster indices by volatility: lowest vol → first label
        sorted_idx = np.argsort(vols)
        return {int(sorted_idx[i]): self.labels[i] for i in range(self.k)}

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """Fit the regime model on historical OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with DatetimeIndex.

        Returns
        -------
        self
        """
        returns = self._log_returns(df)
        windows, indices = window_lift(returns.values, self.window_size, self.stride)

        if len(windows) < self.k:
            raise ValueError(
                f"Not enough data: got {len(windows)} windows for k={self.k}. "
                f"Need at least {self.k} windows. "
                f"Data has {len(returns)} returns, window_size={self.window_size}, stride={self.stride}."
            )

        logger.info(
            "Fitting regime detector: %d windows (size=%d, stride=%d, k=%d)",
            len(windows), self.window_size, self.stride, self.k,
        )

        self.model.fit(windows)
        self._label_map = self._build_label_map()
        self._fitted = True

        logger.info(
            "Regime detector fitted in %d iterations. Label map: %s",
            self.model.n_iter_, self._label_map,
        )
        return self

    def detect(self, df: pd.DataFrame) -> str:
        """Classify the current regime from the most recent bars.

        Uses the last ``window_size`` bars to form a single distribution,
        then predicts its cluster.

        Parameters
        ----------
        df : pd.DataFrame
            Recent OHLCV data (at least ``window_size + 1`` bars).

        Returns
        -------
        str
            Regime label (e.g. ``"low_vol_trending"``).
        """
        if not self._fitted:
            raise ValueError("Model has not been fit yet.")

        returns = self._log_returns(df)

        if len(returns) < self.window_size:
            logger.warning(
                "Not enough data for detection (%d returns, need %d). "
                "Returning most conservative regime.",
                len(returns), self.window_size,
            )
            return self.labels[-1]  # most conservative (highest vol label)

        window = returns.values[-self.window_size:]
        cluster_id = self.model.predict([window])[0]
        return self._label_map[cluster_id]

    def label_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label an entire historical DataFrame with regime assignments.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Columns: ``returns``, ``cumulative_returns``, ``regime``
            (string labels), ``regime_id`` (int cluster IDs).
        """
        if not self._fitted:
            raise ValueError("Model has not been fit yet.")

        returns = self._log_returns(df)
        windows, indices = window_lift(returns.values, self.window_size, self.stride)
        assignments = self.model.predict(windows)

        result = reconstruct(returns, indices, assignments)
        result["regime_label"] = result["regime"].map(
            lambda x: self._label_map.get(int(x), "unknown") if not np.isnan(x) else "unknown"
        )
        return result

    def quality_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute cluster quality metrics on given data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data.

        Returns
        -------
        dict
            Keys: ``within_mmd``, ``between_mmd``, ``separation_ratio``.
        """
        if not self._fitted:
            raise ValueError("Model has not been fit yet.")

        returns = self._log_returns(df)
        windows, indices = window_lift(returns.values, self.window_size, self.stride)
        assignments = self.model.predict(windows)

        return cluster_quality(
            windows, assignments,
            gamma=self.gamma, random_state=self.random_state,
        )

    def summary(self, df: pd.DataFrame) -> str:
        """Return a printable summary of regime detection results."""
        history = self.label_history(df)
        quality = self.quality_scores(df)

        lines = ["Regime Detection Summary", "=" * 40]

        # Centroid stats
        vols = self.model.centroid_volatilities()
        sorted_idx = np.argsort(vols)
        for i, idx in enumerate(sorted_idx):
            label = self.labels[i]
            count = (history["regime"] == idx).sum()
            pct = count / len(history) * 100 if len(history) > 0 else 0
            lines.append(
                f"  {label:25s}: {count:5d} bars ({pct:5.1f}%) | "
                f"centroid vol={vols[idx]:.6f}"
            )

        lines.append(f"\nCluster Quality (MMD)")
        lines.append(f"  Within-cluster:   {quality['within_mmd']:.6f}")
        lines.append(f"  Between-cluster:  {quality['between_mmd']:.6f}")
        lines.append(f"  Separation ratio: {quality['separation_ratio']:.2f}")
        lines.append(f"\nConverged in {self.model.n_iter_} iterations")

        # Current regime
        current = self.detect(df)
        lines.append(f"Current regime: {current}")

        return "\n".join(lines)
