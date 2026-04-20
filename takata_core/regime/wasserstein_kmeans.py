"""Wasserstein k-means clustering for market regime detection.

Based on: Horvath, Issa & Muguruza (2021), "Clustering Market Regimes
using the Wasserstein Distance", arXiv:2110.11848.

Reference implementation: https://github.com/NickS785/WKmeans
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel

logger = logging.getLogger(__name__)


class WassersteinKMeans:
    """K-means clustering in the space of probability distributions.

    Instead of computing Euclidean distances between point observations,
    this algorithm uses the 1-Wasserstein (earth mover's) distance between
    empirical distributions and the Wasserstein barycenter as the cluster
    centroid.

    Parameters
    ----------
    k : int
        Number of clusters (regimes).
    max_iter : int
        Maximum iterations before stopping.
    tol : float
        Convergence tolerance on MMD loss change.
    gamma : float
        RBF kernel bandwidth for MMD computation.
    mmd_pairs : int
        Number of random sample pairs for MMD estimation per cluster.
    sample_size : int
        Sample size drawn from each distribution for MMD computation.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        k: int = 3,
        max_iter: int = 50,
        tol: float = 1e-4,
        gamma: float = 1.0,
        mmd_pairs: int = 10,
        sample_size: int = 50,
        random_state: Optional[int] = None,
    ):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma
        self.mmd_pairs = mmd_pairs
        self.sample_size = sample_size
        self.random_state = random_state

        self.centroids: Optional[List[np.ndarray]] = None
        self.assignments: Optional[List[int]] = None
        self.labels_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.loss_history_: List[float] = []

    def _compute_barycenter(self, distributions: List[np.ndarray]) -> np.ndarray:
        """Wasserstein barycenter via quantile averaging.

        For 1-D distributions, the Wasserstein barycenter is the
        element-wise mean of the sorted (quantile) representations.
        """
        sorted_dists = [np.sort(d) for d in distributions]
        min_len = min(len(d) for d in sorted_dists)
        trimmed = [d[:min_len] for d in sorted_dists]
        return np.mean(trimmed, axis=0)

    def _compute_mmd(self, x: np.ndarray, y: np.ndarray) -> float:
        """Maximum Mean Discrepancy with RBF kernel."""
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        K_xx = rbf_kernel(x, x, gamma=self.gamma)
        K_yy = rbf_kernel(y, y, gamma=self.gamma)
        K_xy = rbf_kernel(x, y, gamma=self.gamma)

        n, m = x.shape[0], y.shape[0]

        mmd_sq = (
            (K_xx.sum() - np.trace(K_xx)) / max(n * (n - 1), 1)
            + (K_yy.sum() - np.trace(K_yy)) / max(m * (m - 1), 1)
            - 2 * K_xy.mean()
        )
        return mmd_sq

    def _cluster_self_similarity(
        self, cluster_samples: List[np.ndarray], rng: np.random.Generator
    ) -> float:
        """Median MMD between random pairs within a cluster."""
        n = len(cluster_samples)
        if n < 2:
            return 0.0

        mmds = []
        for _ in range(self.mmd_pairs):
            i, j = rng.choice(n, size=2, replace=False)
            x = cluster_samples[i][: self.sample_size]
            y = cluster_samples[j][: self.sample_size]
            mmds.append(self._compute_mmd(x, y))

        return float(np.median(mmds))

    def fit(self, distributions: List[np.ndarray]) -> "WassersteinKMeans":
        """Fit the model on a list of empirical distributions.

        Parameters
        ----------
        distributions : list of 1-D np.ndarray
            Each element is a window of return observations treated as
            an empirical distribution.

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)
        n = len(distributions)

        if n < self.k:
            raise ValueError(
                f"Need at least k={self.k} distributions, got {n}"
            )

        # Initialize centroids randomly
        init_idx = rng.choice(n, size=self.k, replace=False)
        self.centroids = [distributions[i].copy() for i in init_idx]

        prev_loss = float("inf")
        self.loss_history_ = []

        for iteration in range(self.max_iter):
            # ── Assignment step ──
            assignments = []
            for dist in distributions:
                dists = [wasserstein_distance(dist, c) for c in self.centroids]
                assignments.append(int(np.argmin(dists)))

            # ── Group into clusters ──
            clusters: List[List[np.ndarray]] = [[] for _ in range(self.k)]
            for idx, cid in enumerate(assignments):
                clusters[cid].append(distributions[idx])

            # ── Update step ──
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    new_centroids.append(self._compute_barycenter(cluster))
                else:
                    # Reinitialize empty cluster
                    new_centroids.append(distributions[rng.integers(n)].copy())

            # ── Convergence check via MMD loss ──
            non_trivial = [c for c in clusters if len(c) >= 2]
            if non_trivial:
                loss = float(
                    np.mean(
                        [self._cluster_self_similarity(c, rng) for c in non_trivial]
                    )
                )
            else:
                loss = 0.0

            self.loss_history_.append(loss)
            logger.debug("Iteration %d, MMD loss: %.6f", iteration + 1, loss)

            if abs(prev_loss - loss) < self.tol:
                logger.info(
                    "Converged at iteration %d (loss=%.6f)", iteration + 1, loss
                )
                self.centroids = new_centroids
                break

            self.centroids = new_centroids
            prev_loss = loss

        # Final assignment
        self.assignments = []
        for dist in distributions:
            dists = [wasserstein_distance(dist, c) for c in self.centroids]
            self.assignments.append(int(np.argmin(dists)))

        self.labels_ = np.array(self.assignments)
        self.n_iter_ = iteration + 1
        return self

    def predict(self, distributions: List[np.ndarray]) -> np.ndarray:
        """Assign new distributions to the nearest centroid.

        Parameters
        ----------
        distributions : list of 1-D np.ndarray

        Returns
        -------
        np.ndarray of int
            Cluster assignments.
        """
        if self.centroids is None:
            raise ValueError("Model has not been fit yet.")

        labels = []
        for dist in distributions:
            dists = [wasserstein_distance(dist, c) for c in self.centroids]
            labels.append(int(np.argmin(dists)))
        return np.array(labels)

    def centroid_volatilities(self) -> np.ndarray:
        """Return the standard deviation of each centroid distribution.

        Useful for sorting clusters by volatility to assign human-readable
        regime labels (low-vol, high-vol, crisis).
        """
        if self.centroids is None:
            raise ValueError("Model has not been fit yet.")
        return np.array([np.std(c) for c in self.centroids])
