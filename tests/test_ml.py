"""Unit tests for ML layer."""

from datetime import datetime

import numpy as np
import pytest

from takata_engine.ml.feature_store import (
    FEATURE_COLUMNS,
    build_training_data,
    signal_to_features,
)
from takata_engine.ml.signal_learner import SignalLearner
from takata_engine.ml.portfolio_feedback import PortfolioFeedback
from takata_engine.signals.position import Position
from takata_engine.signals.signal_generator import Signal


def _make_signal(direction="long", strength=0.6, price=100.0, rsi=55.0, macd=0.5) -> Signal:
    return Signal(
        timestamp=datetime(2025, 1, 2, 10, 0),
        instrument="MES", direction=direction, price=price,
        stop=price - 5, target=price + 10, regime="calm",
        strength=strength, reasons=["ema_bullish_cross", "rsi_rising"],
        indicators={
            "ema_fast": price + 1, "ema_slow": price - 1,
            "rsi": rsi, "macd": macd, "macd_signal": macd - 0.1,
            "macd_hist": 0.1, "adx": 30, "plus_di": 25, "minus_di": 15,
            "atr": 5, "bb_upper": price + 10, "bb_middle": price,
            "bb_lower": price - 10, "vwap": price - 1, "close": price,
        },
    )


def _make_position(entry=100.0, exit_p=105.0, side="long") -> Position:
    p = Position("MES", side, entry, datetime(2025, 1, 2, 10, 0),
                 1.0, entry - 5, entry + 10, "calm")
    p.close(exit_p, datetime(2025, 1, 2, 11, 0), "target_hit")
    return p


class TestFeatureStore:
    def test_signal_to_features(self):
        sig = _make_signal()
        features = signal_to_features(sig)
        assert "rsi" in features
        assert "ema_spread" in features
        assert features["direction"] == 1.0

    def test_build_training_data(self):
        signals = [_make_signal() for _ in range(10)]
        positions = [_make_position(exit_p=105) for _ in range(5)] + \
                    [_make_position(exit_p=95) for _ in range(5)]
        df = build_training_data(signals, positions)
        assert len(df) > 0
        assert "profitable" in df.columns

    def test_feature_columns_present(self):
        sig = _make_signal()
        features = signal_to_features(sig)
        for col in FEATURE_COLUMNS:
            assert col in features, f"Missing feature: {col}"


class TestSignalLearner:
    def test_train_with_enough_data(self, tmp_path):
        # Generate diverse training data — 100 samples, ~50/50 win/loss
        signals = []
        positions = []
        rng = np.random.default_rng(42)

        for i in range(100):
            rsi_val = rng.uniform(20, 80)
            macd_val = rng.uniform(-2, 2)
            price = 100 + rng.uniform(-10, 10)
            sig = _make_signal(
                direction="long" if rsi_val > 50 else "short",
                strength=rng.uniform(0.3, 0.9),
                price=price, rsi=rsi_val, macd=macd_val,
            )
            ts = datetime(2025, 1, 2 + i // 60, 10, i % 60)
            sig.timestamp = ts
            signals.append(sig)

            # Alternating wins/losses with some noise
            win = (rsi_val > 50 and macd_val > 0) or rng.random() > 0.6
            exit_p = price + 5 if win else price - 3
            p = Position("MES", sig.direction, price, ts,
                         1.0, price - 5, price + 10, "calm")
            p.close(exit_p, datetime(2025, 1, 2 + i // 60, 11, i % 60), "target_hit")
            positions.append(p)

        learner = SignalLearner(model_path=str(tmp_path / "test_model.json"))
        stats = learner.train(signals, positions)

        assert "error" not in stats
        assert stats["samples"] == 100
        assert stats["accuracy"] >= 0.4  # at least better than pure chance
        assert learner.is_trained
        assert learner.feature_importances_ is not None

    def test_predict_without_training(self):
        learner = SignalLearner(model_path="/nonexistent/model.json")
        sig = _make_signal()
        result = learner.predict(sig)
        assert result["recommendation"] == "no_model"

    def test_summary(self):
        learner = SignalLearner()
        summary = learner.summary()
        assert "not trained" in summary.lower()


class TestPortfolioFeedback:
    def test_record_and_analyze(self, tmp_path):
        import takata_engine.ml.portfolio_feedback as pf
        pf.FEEDBACK_DIR = tmp_path

        fb = PortfolioFeedback()
        fb.feedback_path = tmp_path / "test_outcomes.jsonl"

        for i in range(10):
            fb.record_outcome(
                client_id=f"client_{i}",
                risk_profile="moderate",
                allocation={"US Equity": 0.6, "Bonds": 0.3, "Gold": 0.1},
                period_return=0.05 + (i * 0.01),
                period_volatility=0.12,
                period_sharpe=0.5 + (i * 0.1),
                max_drawdown=0.08,
                benchmark_return=0.04,
            )

        assert len(fb._data) == 10

        insights = fb.analyze()
        assert "by_risk_profile" in insights
        assert "moderate" in insights["by_risk_profile"]
        assert insights["by_risk_profile"]["moderate"]["observations"] == 10
