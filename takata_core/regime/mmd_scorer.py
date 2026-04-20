"""Maximum Mean Discrepancy scoring for cluster quality validation."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def mmd_score(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """Compute MMD² between two sample sets using RBF kernel.

    Parameters
    ----------
    x, y : np.ndarray
        1-D or 2-D sample arrays.
    gamma : float
        RBF kernel bandwidth.

    Returns
    -------
    float
        MMD² value. Lower means more similar distributions.
    """
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y

    K_xx = rbf_kernel(x, x, gamma=gamma)
    K_yy = rbf_kernel(y, y, gamma=gamma)
    K_xy = rbf_kernel(x, y, gamma=gamma)

    n, m = x.shape[0], y.shape[0]

    mmd_sq = (
        (K_xx.sum() - np.trace(K_xx)) / max(n * (n - 1), 1)
        + (K_yy.sum() - np.trace(K_yy)) / max(m * (m - 1), 1)
        - 2 * K_xy.mean()
    )
    return float(mmd_sq)


def cluster_quality(
    distributions: List[np.ndarray],
    assignments: np.ndarray | List[int],
    gamma: float = 1.0,
    n_pairs: int = 10,
    sample_size: int = 50,
    random_state: int | None = None,
) -> Dict[str, float]:
    """Evaluate clustering quality via within- and between-cluster MMD.

    Parameters
    ----------
    distributions : list of np.ndarray
        The windowed distributions that were clustered.
    assignments : array-like of int
        Cluster label for each distribution.
    gamma : float
        RBF kernel bandwidth.
    n_pairs : int
        Number of random pairs to sample for each metric.
    sample_size : int
        Max samples drawn from each distribution.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    dict
        ``within_mmd``: mean within-cluster MMD (lower is better).
        ``between_mmd``: mean between-cluster MMD (higher is better).
        ``separation_ratio``: between / within (higher is better).
    """
    rng = np.random.default_rng(random_state)
    assignments = np.asarray(assignments)
    labels = np.unique(assignments)

    # Group distributions by cluster
    clusters: Dict[int, List[np.ndarray]] = {l: [] for l in labels}
    for dist, label in zip(distributions, assignments):
        clusters[label].append(dist[:sample_size])

    # Within-cluster MMD
    within_scores = []
    for label in labels:
        members = clusters[label]
        if len(members) < 2:
            continue
        for _ in range(n_pairs):
            i, j = rng.choice(len(members), size=2, replace=False)
            within_scores.append(mmd_score(members[i], members[j], gamma))

    within_mmd = float(np.mean(within_scores)) if within_scores else 0.0

    # Between-cluster MMD
    between_scores = []
    if len(labels) >= 2:
        for _ in range(n_pairs * len(labels)):
            l1, l2 = rng.choice(labels, size=2, replace=False)
            i = rng.integers(len(clusters[l1]))
            j = rng.integers(len(clusters[l2]))
            between_scores.append(
                mmd_score(clusters[l1][i], clusters[l2][j], gamma)
            )

    between_mmd = float(np.mean(between_scores)) if between_scores else 0.0

    separation = between_mmd / within_mmd if within_mmd > 0 else float("inf")

    return {
        "within_mmd": within_mmd,
        "between_mmd": between_mmd,
        "separation_ratio": separation,
    }
