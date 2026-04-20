"""Unit tests for bar builder."""

from datetime import datetime

import pytest

from takata_engine.data.bar_builder import BarBuilder, Tick


class TestBarBuilderTime:
    def test_emits_bar_on_period_change(self):
        builder = BarBuilder(mode="time", interval="1min")
        ticks = [
            Tick(datetime(2025, 1, 2, 9, 30, 0), 100.0, 10),
            Tick(datetime(2025, 1, 2, 9, 30, 30), 101.0, 20),
            Tick(datetime(2025, 1, 2, 9, 31, 0), 102.0, 15),  # new period
        ]
        bars = [builder.process_tick(t) for t in ticks]
        assert bars[0] is None
        assert bars[1] is None
        assert bars[2] is not None
        assert bars[2].open == 100.0
        assert bars[2].close == 101.0
        assert bars[2].high == 101.0
        assert bars[2].low == 100.0
        assert bars[2].volume == 30

    def test_flush(self):
        builder = BarBuilder(mode="time", interval="5min")
        builder.process_tick(Tick(datetime(2025, 1, 2, 9, 30, 0), 50.0, 5))
        bar = builder.flush()
        assert bar is not None
        assert bar.close == 50.0


class TestBarBuilderTick:
    def test_emits_after_n_ticks(self):
        builder = BarBuilder(mode="tick", interval=3)
        results = []
        for i in range(6):
            bar = builder.process_tick(
                Tick(datetime(2025, 1, 2, 9, 30, i), 100.0 + i, 1)
            )
            results.append(bar)
        # Bar emitted at tick 3 and tick 6
        assert results[2] is not None
        assert results[5] is not None
        assert results[0] is None
