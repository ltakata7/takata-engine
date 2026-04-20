"""Regime model auto-tuner — tracks detection accuracy and suggests parameter updates."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from takata_core.backtest.backtester import Backtester
from takata_core.regime import RegimeDetector

logger = logging.getLogger(__name__)


class RegimeTuner:
    """Evaluates and tunes regime detection parameters.

    Runs backtests across different regime configs and identifies
    which settings produce the best risk-adjusted returns.

    Parameters
    ----------
    instrument : str
        Instrument symbol.
    feed_path : str
        Path to historical data.
    """

    def __init__(self, instrument: str = "MES", feed_path: str = "sample_data/mes_5min.csv"):
        self.instrument = instrument
        self.feed_path = feed_path

    def evaluate_configs(
        self,
        configs: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """Test multiple regime detection configurations.

        Parameters
        ----------
        configs : list of dict, optional
            Each dict has keys: ``k``, ``window_size``, ``stride``.
            If None, uses a default grid.

        Returns
        -------
        pd.DataFrame
            Results ranked by Sharpe ratio.
        """
        if configs is None:
            configs = [
                {"regime.k": k, "regime.window_size": ws, "regime.stride": s}
                for k in [2, 3, 4]
                for ws in [20, 30, 50]
                for s in [5, 10]
            ]

        bt = Backtester(
            instrument=self.instrument,
            feed_path=self.feed_path,
        )

        results = []
        for i, config in enumerate(configs):
            logger.info("[%d/%d] Testing config: %s", i + 1, len(configs), config)
            try:
                result = bt.run(config_overrides=config)
                m = result.metrics
                results.append({
                    **config,
                    "sharpe": m.get("sharpe_ratio", 0),
                    "total_pnl": m.get("total_pnl", 0),
                    "win_rate": m.get("win_rate", 0),
                    "max_drawdown": m.get("max_drawdown", 0),
                    "trades": m.get("total_trades", 0),
                    "profit_factor": m.get("profit_factor", 0),
                })
            except Exception as e:
                logger.warning("Config failed: %s — %s", config, e)

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

        return df

    def recommend(self, results: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Get the best regime configuration.

        Returns
        -------
        dict
            Best config and its performance metrics.
        """
        if results is None:
            results = self.evaluate_configs()

        if results.empty:
            return {"error": "No valid configurations found"}

        best = results.iloc[0]
        return {
            "best_config": {k: v for k, v in best.items() if k.startswith("regime.")},
            "sharpe": float(best.get("sharpe", 0)),
            "total_pnl": float(best.get("total_pnl", 0)),
            "win_rate": float(best.get("win_rate", 0)),
            "trades": int(best.get("trades", 0)),
            "total_configs_tested": len(results),
        }
