"""Unit tests for Wasserstein k-means clustering."""

import numpy as np
import pytest

from takata_core.regime.wasserstein_kmeans import WassersteinKMeans
from takata_core.regime.window import window_lift, reconstruct
from takata_core.regime.mmd_scorer import mmd_score, cluster_quality


class TestWassersteinKMeans:
    @pytest.fixture
    def three_distributions(self):
        """3 clearly separated distributions: N(0,0.5), N(0,2), N(3,1)."""
        rng = np.random.default_rng(42)
        dists = []
        labels = []
        for _ in range(20):
            dists.append(rng.normal(0, 0.5, 100))   # low vol, centered
            labels.append(0)
        for _ in range(20):
            dists.append(rng.normal(0, 2.0, 100))    # high vol, centered
            labels.append(1)
        for _ in range(20):
            dists.append(rng.normal(3, 1.0, 100))    # shifted mean
            labels.append(2)
        return dists, labels

    def test_fit_returns_self(self, three_distributions):
        dists, _ = three_distributions
        model = WassersteinKMeans(k=3, random_state=42)
        result = model.fit(dists)
        assert result is model

    def test_recovers_three_clusters(self, three_distributions):
        dists, true_labels = three_distributions
        model = WassersteinKMeans(k=3, random_state=42)
        model.fit(dists)

        # Check that each true group maps to a single predicted cluster
        predicted = model.labels_
        for group_start in range(0, 60, 20):
            group = predicted[group_start : group_start + 20]
            # All 20 samples in each group should have the same label
            assert len(set(group)) == 1, (
                f"Group starting at {group_start} has mixed labels: {set(group)}"
            )

        # All three groups should map to different clusters
        group_labels = {predicted[0], predicted[20], predicted[40]}
        assert len(group_labels) == 3

    def test_predict_matches_fit(self, three_distributions):
        dists, _ = three_distributions
        model = WassersteinKMeans(k=3, random_state=42)
        model.fit(dists)
        predicted = model.predict(dists)
        np.testing.assert_array_equal(model.labels_, predicted)

    def test_reproducibility(self, three_distributions):
        dists, _ = three_distributions
        m1 = WassersteinKMeans(k=3, random_state=123).fit(dists)
        m2 = WassersteinKMeans(k=3, random_state=123).fit(dists)
        np.testing.assert_array_equal(m1.labels_, m2.labels_)

    def test_predict_before_fit_raises(self):
        model = WassersteinKMeans(k=3)
        with pytest.raises(ValueError, match="not been fit"):
            model.predict([np.array([1, 2, 3])])

    def test_centroid_volatilities(self, three_distributions):
        dists, _ = three_distributions
        model = WassersteinKMeans(k=3, random_state=42).fit(dists)
        vols = model.centroid_volatilities()
        assert len(vols) == 3
        assert all(v > 0 for v in vols)

    def test_too_few_distributions(self):
        model = WassersteinKMeans(k=3)
        with pytest.raises(ValueError, match="at least"):
            model.fit([np.array([1, 2, 3])])


class TestWindowLift:
    def test_basic(self):
        series = np.arange(100, dtype=float)
        windows, indices = window_lift(series, window_size=10, stride=5)
        assert len(windows) == 19  # (100 - 10) / 5 + 1
        assert windows[0].shape == (10,)
        assert indices[0] == (0, 10)
        assert indices[1] == (5, 15)

    def test_stride_equals_window(self):
        series = np.arange(30, dtype=float)
        windows, indices = window_lift(series, window_size=10, stride=10)
        assert len(windows) == 3


class TestReconstruct:
    def test_output_shape(self):
        import pandas as pd
        returns = pd.Series(np.random.randn(50), index=pd.date_range("2025-01-01", periods=50, freq="D"))
        indices = [(0, 10), (5, 15), (10, 20)]
        assignments = [0, 1, 0]
        result = reconstruct(returns, indices, assignments)
        assert len(result) == 50
        assert "regime" in result.columns
        assert "cumulative_returns" in result.columns


class TestMMDScorer:
    def test_same_distribution(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        score = mmd_score(x, x.copy())
        assert score < 0.01  # should be near 0

    def test_different_distributions(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(5, 1, 100)
        score = mmd_score(x, y)
        assert score > 0.1  # should be meaningfully different

    def test_cluster_quality_structure(self):
        rng = np.random.default_rng(42)
        dists = [rng.normal(0, 1, 50) for _ in range(10)] + [rng.normal(5, 1, 50) for _ in range(10)]
        assignments = [0] * 10 + [1] * 10
        result = cluster_quality(dists, assignments, random_state=42)
        assert "within_mmd" in result
        assert "between_mmd" in result
        assert "separation_ratio" in result
        assert result["separation_ratio"] > 1.0  # between > within
