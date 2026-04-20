"""Signal quality learner — XGBoost model that predicts profitable trades.

Learns which indicator combinations and market conditions produce winning
signals. Improves over time as more trade outcomes are recorded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from takata_core.ml.feature_store import (
    FEATURE_COLUMNS,
    build_training_data,
    signal_to_features,
)
from takata_core.signals.position import Position
from takata_core.signals.signal_generator import Signal

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class SignalLearner:
    """XGBoost model that learns to predict signal quality.

    The model is trained on historical signal outcomes (profitable vs not)
    and predicts the probability that a new signal will be profitable.

    Parameters
    ----------
    model_path : str, optional
        Path to save/load the trained model.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else MODEL_DIR / "signal_quality.json"
        self.model = None
        self.is_trained = False
        self.feature_importances_: Optional[Dict[str, float]] = None
        self.training_stats: Dict[str, Any] = {}

    def train(
        self,
        signals: List[Signal],
        positions: List[Position],
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Train the model on historical signal outcomes.

        Parameters
        ----------
        signals : list of Signal
            Historical signals with indicator snapshots.
        positions : list of Position
            Closed trades with P&L outcomes.
        test_size : float
            Fraction of data to hold out for validation.

        Returns
        -------
        dict
            Training metrics: accuracy, precision, recall, feature importances.
        """
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        df = build_training_data(signals, positions)

        if len(df) < 20:
            logger.warning("Not enough training data (%d samples, need 20+)", len(df))
            return {"error": "insufficient_data", "samples": len(df)}

        X = df[FEATURE_COLUMNS].fillna(0)
        y = df["profitable"]

        if y.nunique() < 2:
            logger.warning("Only one class in data (all winners or all losers)")
            return {"error": "single_class", "samples": len(df), "win_rate": float(y.mean())}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y,
        )

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))

        # Feature importances
        importances = dict(zip(FEATURE_COLUMNS, self.model.feature_importances_.tolist()))
        self.feature_importances_ = dict(sorted(importances.items(), key=lambda x: -x[1]))

        self.is_trained = True
        self.training_stats = {
            "samples": len(df),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "win_rate_actual": float(y.mean()),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "feature_importances": self.feature_importances_,
        }

        # Save model
        self._save()
        logger.info(
            "Signal learner trained: %d samples, accuracy=%.1f%%, precision=%.1f%%",
            len(df), accuracy * 100, precision * 100,
        )

        return self.training_stats

    def predict(self, signal: Signal) -> Dict[str, Any]:
        """Predict the quality of a new signal.

        Returns
        -------
        dict
            ``probability``: 0-1 chance of being profitable.
            ``recommendation``: "strong", "moderate", "weak", "avoid".
            ``features``: the feature values used.
        """
        if not self.is_trained:
            self._load()
            if not self.is_trained:
                return {"probability": 0.5, "recommendation": "no_model", "features": {}}

        features = signal_to_features(signal)
        X = pd.DataFrame([{k: features.get(k, 0) for k in FEATURE_COLUMNS}])

        prob = float(self.model.predict_proba(X)[0, 1])

        if prob >= 0.7:
            rec = "strong"
        elif prob >= 0.55:
            rec = "moderate"
        elif prob >= 0.4:
            rec = "weak"
        else:
            rec = "avoid"

        return {
            "probability": round(prob, 3),
            "recommendation": rec,
            "features": {k: round(v, 4) for k, v in features.items() if k in FEATURE_COLUMNS},
        }

    def _save(self) -> None:
        MODEL_DIR.mkdir(exist_ok=True)
        if self.model:
            self.model.save_model(str(self.model_path))
            # Save stats separately
            stats_path = self.model_path.with_suffix(".stats.json")
            with open(stats_path, "w") as f:
                json.dump(self.training_stats, f, indent=2, default=str)

    def _load(self) -> None:
        if self.model_path.exists():
            import xgboost as xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(self.model_path))
            self.is_trained = True

            stats_path = self.model_path.with_suffix(".stats.json")
            if stats_path.exists():
                with open(stats_path) as f:
                    self.training_stats = json.load(f)
                    self.feature_importances_ = self.training_stats.get("feature_importances")

            logger.info("Signal learner loaded from %s", self.model_path)

    def summary(self) -> str:
        """Return training summary."""
        if not self.training_stats:
            return "Signal learner not trained yet."

        s = self.training_stats
        lines = [
            "Signal Quality Model",
            "=" * 50,
            f"  Training samples:  {s.get('samples', 0)}",
            f"  Actual win rate:   {s.get('win_rate_actual', 0):.1%}",
            f"  Model accuracy:    {s.get('accuracy', 0):.1%}",
            f"  Precision:         {s.get('precision', 0):.1%}",
            f"  Recall:            {s.get('recall', 0):.1%}",
            "",
            "  Top Features:",
        ]
        if self.feature_importances_:
            for feat, imp in list(self.feature_importances_.items())[:5]:
                bar = "#" * int(imp * 50)
                lines.append(f"    {feat:20s} {imp:.3f}  {bar}")

        return "\n".join(lines)
