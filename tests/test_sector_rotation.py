"""Unit tests for sector rotation model."""

import pytest

from takata_engine.sector_rotation.model import SectorRotationModel, SECTOR_ETFS


class TestSectorRotation:
    def test_all_sectors_covered(self):
        assert len(SECTOR_ETFS) == 11

    def test_analyze_returns_scores(self):
        model = SectorRotationModel(period="6mo")
        scores = model.analyze()
        assert len(scores) > 0
        # Each score should have required fields
        for s in scores:
            assert s.etf in SECTOR_ETFS.values()
            assert 0 <= s.momentum_score <= 100
            assert s.recommendation in ("OVERWEIGHT", "EQUAL WEIGHT", "UNDERWEIGHT")
            assert s.allocation_pct > 0

    def test_allocations_sum_to_100(self):
        model = SectorRotationModel(period="6mo")
        scores = model.analyze()
        total = sum(s.allocation_pct for s in scores)
        assert abs(total - 100.0) < 1.0  # within rounding
