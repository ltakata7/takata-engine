"""Walk-forward optimization — rolling in-sample/out-of-sample validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from takata_core.backtest.backtester import Backtester, BacktestResult
from takata_core.backtest.metrics import compute_all, format_report

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Result of a single walk-forward window."""

    window_index: int
    in_sample_start: str
    in_sample_end: str
    out_of_sample_start: str
    out_of_sample_end: str
    best_params: Dict[str, Any]
    in_sample_metrics: Dict[str, Any]
    out_of_sample_metrics: Dict[str, Any]


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward optimization result."""

    windows: List[WindowResult]
    aggregate_oos_positions: List[Any]  # Position objects
    aggregate_oos_metrics: Dict[str, Any]

    def summary(self) -> str:
        lines = [
            "Walk-Forward Optimization Results",
            "=" * 60,
            f"  Windows:          {len(self.windows)}",
            "",
        ]

        for w in self.windows:
            lines.append(
                f"  Window {w.window_index + 1}: "
                f"IS [{w.in_sample_start} → {w.in_sample_end}] "
                f"OOS [{w.out_of_sample_start} → {w.out_of_sample_end}]"
            )
            is_sharpe = w.in_sample_metrics.get("sharpe_ratio", 0)
            oos_sharpe = w.out_of_sample_metrics.get("sharpe_ratio", 0)
            oos_trades = w.out_of_sample_metrics.get("total_trades", 0)
            oos_pnl = w.out_of_sample_metrics.get("total_pnl", 0)
            lines.append(
                f"    Best params: {w.best_params}"
            )
            lines.append(
                f"    IS Sharpe={is_sharpe:.2f} | "
                f"OOS Sharpe={oos_sharpe:.2f}, trades={oos_trades}, P&L={oos_pnl:,.2f}"
            )
            lines.append("")

        lines.append("─" * 60)
        lines.append("Aggregate Out-of-Sample Performance:")
        lines.append(format_report(self.aggregate_oos_metrics, title=""))
        return "\n".join(lines)


class WalkForwardOptimizer:
    """Rolling walk-forward optimization.

    Splits data into sequential [in-sample | out-of-sample] windows.
    For each window, optimizes parameters on IS data and validates on OOS.

    Parameters
    ----------
    instrument : str
        Instrument symbol.
    feed_path : str
        Path to CSV data.
    in_sample_bars : int
        Number of bars for in-sample optimization.
    out_of_sample_bars : int
        Number of bars for out-of-sample validation.
    param_grid : dict
        Parameter grid for optimization (dot-notation keys → value lists).
    initial_capital : float
        Starting capital.
    regime_seed : int
        Random seed for regime detector.
    min_strength : float
        Minimum signal strength.
    """

    def __init__(
        self,
        instrument: str = "MES",
        feed_path: str = "sample_data/mes_5min.csv",
        in_sample_bars: int = 500,
        out_of_sample_bars: int = 200,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        initial_capital: float = 10000.0,
        regime_seed: int = 42,
        min_strength: float = 0.5,
    ):
        self.instrument = instrument
        self.feed_path = feed_path
        self.in_sample_bars = in_sample_bars
        self.out_of_sample_bars = out_of_sample_bars
        self.param_grid = param_grid or {
            "risk.stop_atr_mult": [1.0, 1.5, 2.0],
            "risk.target_atr_mult": [1.5, 2.0, 3.0],
        }
        self.initial_capital = initial_capital
        self.regime_seed = regime_seed
        self.min_strength = min_strength

    def run(self) -> WalkForwardResult:
        """Execute walk-forward optimization.

        Returns
        -------
        WalkForwardResult
        """
        from takata_core.data import CSVFeed

        feed = CSVFeed(self.feed_path)
        df = feed.load()

        window_size = self.in_sample_bars + self.out_of_sample_bars
        n_windows = (len(df) - self.in_sample_bars) // self.out_of_sample_bars

        if n_windows < 1:
            raise ValueError(
                f"Not enough data for walk-forward: {len(df)} bars, "
                f"need at least {window_size} (IS={self.in_sample_bars} + OOS={self.out_of_sample_bars})"
            )

        logger.info(
            "Walk-forward: %d windows, IS=%d bars, OOS=%d bars",
            n_windows, self.in_sample_bars, self.out_of_sample_bars,
        )

        windows: List[WindowResult] = []
        all_oos_positions = []

        for i in range(n_windows):
            is_start = i * self.out_of_sample_bars
            is_end = is_start + self.in_sample_bars
            oos_start = is_end
            oos_end = min(oos_start + self.out_of_sample_bars, len(df))

            if oos_end <= oos_start:
                break

            is_df = df.iloc[is_start:is_end]
            oos_df = df.iloc[oos_start:oos_end]

            logger.info(
                "  Window %d: IS [%s → %s], OOS [%s → %s]",
                i + 1, is_df.index[0], is_df.index[-1],
                oos_df.index[0], oos_df.index[-1],
            )

            # Optimize on in-sample
            bt = Backtester(
                instrument=self.instrument,
                feed_path=self.feed_path,
                initial_capital=self.initial_capital,
                regime_seed=self.regime_seed,
                min_strength=self.min_strength,
            )
            is_results = bt.run_sweep(self.param_grid, df=is_df)

            if not is_results:
                continue

            best = is_results[0]
            best_params = best.config_overrides

            # Validate on out-of-sample with best params
            oos_result = bt.run(df=oos_df, config_overrides=best_params)

            windows.append(WindowResult(
                window_index=i,
                in_sample_start=str(is_df.index[0]),
                in_sample_end=str(is_df.index[-1]),
                out_of_sample_start=str(oos_df.index[0]),
                out_of_sample_end=str(oos_df.index[-1]),
                best_params=best_params,
                in_sample_metrics=best.metrics,
                out_of_sample_metrics=oos_result.metrics,
            ))

            all_oos_positions.extend(oos_result.positions)

        # Aggregate OOS metrics
        agg_metrics = compute_all(all_oos_positions, self.initial_capital)

        return WalkForwardResult(
            windows=windows,
            aggregate_oos_positions=all_oos_positions,
            aggregate_oos_metrics=agg_metrics,
        )
