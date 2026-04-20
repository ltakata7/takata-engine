"""Portfolio performance feedback — learns which allocations work best per risk profile."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEEDBACK_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "feedback"


class PortfolioFeedback:
    """Tracks portfolio allocation outcomes and learns optimal allocations.

    Records what was allocated, what happened, and builds a feedback dataset
    that improves future portfolio construction recommendations.
    """

    def __init__(self):
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        self.feedback_path = FEEDBACK_DIR / "allocation_outcomes.jsonl"
        self._data: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.feedback_path.exists():
            with open(self.feedback_path) as f:
                self._data = [json.loads(line) for line in f if line.strip()]

    def record_outcome(
        self,
        client_id: str,
        risk_profile: str,
        allocation: Dict[str, float],
        period_return: float,
        period_volatility: float,
        period_sharpe: float,
        max_drawdown: float,
        benchmark_return: float,
    ) -> None:
        """Record a portfolio allocation outcome for learning.

        Parameters
        ----------
        client_id : str
            Client identifier.
        risk_profile : str
            Risk tolerance level.
        allocation : dict
            Asset class → weight mapping at start of period.
        period_return : float
            Realized return over the measurement period.
        period_volatility : float
            Realized volatility.
        period_sharpe : float
            Realized Sharpe ratio.
        max_drawdown : float
            Max drawdown during period.
        benchmark_return : float
            Benchmark return for comparison.
        """
        record = {
            "client_id": client_id,
            "risk_profile": risk_profile,
            "allocation": allocation,
            "period_return": period_return,
            "period_volatility": period_volatility,
            "period_sharpe": period_sharpe,
            "max_drawdown": max_drawdown,
            "benchmark_return": benchmark_return,
            "alpha": period_return - benchmark_return,
        }

        self._data.append(record)

        with open(self.feedback_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def analyze(self) -> Dict[str, Any]:
        """Analyze allocation outcomes to find patterns.

        Returns
        -------
        dict
            Insights by risk profile: what worked, what didn't.
        """
        if len(self._data) < 5:
            return {"error": "insufficient_data", "samples": len(self._data)}

        df = pd.DataFrame(self._data)

        insights = {
            "total_observations": len(df),
            "by_risk_profile": {},
        }

        for profile, group in df.groupby("risk_profile"):
            avg_return = float(group["period_return"].mean())
            avg_alpha = float(group["alpha"].mean())
            avg_sharpe = float(group["period_sharpe"].mean())
            avg_dd = float(group["max_drawdown"].mean())

            # Find best and worst allocation patterns
            best_idx = group["period_sharpe"].idxmax()
            worst_idx = group["period_sharpe"].idxmin()

            insights["by_risk_profile"][profile] = {
                "observations": len(group),
                "avg_return": round(avg_return, 4),
                "avg_alpha": round(avg_alpha, 4),
                "avg_sharpe": round(avg_sharpe, 2),
                "avg_max_drawdown": round(avg_dd, 4),
                "best_allocation": group.loc[best_idx, "allocation"] if pd.notna(best_idx) else {},
                "best_sharpe": float(group.loc[best_idx, "period_sharpe"]) if pd.notna(best_idx) else 0,
                "worst_allocation": group.loc[worst_idx, "allocation"] if pd.notna(worst_idx) else {},
                "worst_sharpe": float(group.loc[worst_idx, "period_sharpe"]) if pd.notna(worst_idx) else 0,
            }

        return insights

    def suggest_adjustment(self, risk_profile: str, current_allocation: Dict[str, float]) -> Dict[str, Any]:
        """Suggest allocation adjustments based on historical feedback.

        Parameters
        ----------
        risk_profile : str
            Client risk profile.
        current_allocation : dict
            Current asset class weights.

        Returns
        -------
        dict
            Suggested adjustments and rationale.
        """
        insights = self.analyze()

        if "error" in insights:
            return {"suggestion": "no_data", "rationale": "Not enough historical data to make suggestions"}

        profile_data = insights.get("by_risk_profile", {}).get(risk_profile, {})

        if not profile_data or profile_data.get("observations", 0) < 3:
            return {"suggestion": "no_data", "rationale": f"Not enough data for {risk_profile} profile"}

        best_alloc = profile_data.get("best_allocation", {})
        adjustments = {}

        for asset_class, current_weight in current_allocation.items():
            best_weight = best_alloc.get(asset_class, current_weight)
            diff = best_weight - current_weight
            if abs(diff) > 0.03:  # Only suggest if >3% difference
                adjustments[asset_class] = {
                    "current": round(current_weight, 3),
                    "suggested": round(best_weight, 3),
                    "change": round(diff, 3),
                    "direction": "increase" if diff > 0 else "decrease",
                }

        return {
            "suggestion": "adjust" if adjustments else "hold",
            "adjustments": adjustments,
            "rationale": f"Based on {profile_data['observations']} historical outcomes for {risk_profile} profiles. "
                        f"Best historical Sharpe: {profile_data['best_sharpe']:.2f}",
            "avg_alpha": profile_data.get("avg_alpha", 0),
        }
