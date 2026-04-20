"""Backtesting engine — wraps signal generation with metrics and parameter sweeps."""

from __future__ import annotations

import copy
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from takata_core.backtest.metrics import (
    compute_all,
    equity_curve,
    format_report,
    returns_series,
)
from takata_core.config import load_config
from takata_core.config.loader import _deep_merge
from takata_core.data import CSVFeed
from takata_core.regime import RegimeDetector
from takata_core.signals import Signal, SignalGenerator
from takata_core.signals.position import Position

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a single backtest run."""

    positions: List[Position]
    signals: List[Signal]
    metrics: Dict[str, Any]
    equity: pd.Series
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def sharpe(self) -> float:
        return self.metrics.get("sharpe_ratio", 0.0)

    @property
    def total_pnl(self) -> float:
        return self.metrics.get("total_pnl", 0.0)

    def report(self, title: str = "Backtest Report") -> str:
        return format_report(self.metrics, title)


def _set_nested(d: dict, dotted_key: str, value: Any) -> dict:
    """Set a value in a nested dict using dot notation (e.g. 'risk.stop_atr_mult')."""
    keys = dotted_key.split(".")
    current = d
    for k in keys[:-1]:
        current = current.setdefault(k, {})
    current[keys[-1]] = value
    return d


class Backtester:
    """Backtesting engine for the signal pipeline.

    Parameters
    ----------
    instrument : str
        Instrument symbol (``"MES"`` or ``"WDO"``).
    feed_path : str
        Path to CSV data file.
    initial_capital : float
        Starting capital for equity curve.
    regime_seed : int
        Random seed for regime detector.
    min_strength : float
        Minimum signal strength threshold.
    account_risk_per_trade : float
        Max loss per trade in currency units.
    """

    def __init__(
        self,
        instrument: str = "MES",
        feed_path: str = "sample_data/mes_5min.csv",
        initial_capital: float = 10000.0,
        regime_seed: int = 42,
        min_strength: float = 0.5,
        account_risk_per_trade: float = 100.0,
    ):
        self.instrument = instrument
        self.feed_path = feed_path
        self.initial_capital = initial_capital
        self.regime_seed = regime_seed
        self.min_strength = min_strength
        self.account_risk_per_trade = account_risk_per_trade

    def run(
        self,
        df: Optional[pd.DataFrame] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """Execute a single backtest.

        Parameters
        ----------
        df : pd.DataFrame, optional
            OHLCV data. If ``None``, loads from ``feed_path``.
        config_overrides : dict, optional
            Dot-notation overrides (e.g. ``{"risk.stop_atr_mult": 1.5}``).

        Returns
        -------
        BacktestResult
        """
        if df is None:
            feed = CSVFeed(self.feed_path)
            df = feed.load()

        # Build config with overrides
        cfg = load_config(self.instrument)
        if config_overrides:
            override_dict = {}
            for key, val in config_overrides.items():
                _set_nested(override_dict, key, val)
            cfg = _deep_merge(cfg, override_dict)

        # Fit regime detector
        regime_cfg = cfg.get("regime", {})
        detector = RegimeDetector(
            k=regime_cfg.get("k", 3),
            window_size=regime_cfg.get("window_size", 30),
            stride=regime_cfg.get("stride", 5),
            max_iter=regime_cfg.get("max_iter", 50),
            gamma=regime_cfg.get("gamma", 1.0),
            random_state=self.regime_seed,
        )
        detector.fit(df)

        # Run signal generator
        generator = SignalGenerator(
            config=cfg,
            regime_detector=detector,
            account_risk_per_trade=self.account_risk_per_trade,
        )
        signals = generator.run(df, min_strength=self.min_strength)
        positions = generator.position_mgr.closed_positions

        # Compute metrics
        metrics = compute_all(positions, self.initial_capital)
        eq = equity_curve(positions, self.initial_capital)

        return BacktestResult(
            positions=positions,
            signals=signals,
            metrics=metrics,
            equity=eq,
            config_overrides=config_overrides or {},
        )

    def run_sweep(
        self,
        param_grid: Dict[str, List[Any]],
        df: Optional[pd.DataFrame] = None,
    ) -> List[BacktestResult]:
        """Run backtests across a grid of parameter combinations.

        Parameters
        ----------
        param_grid : dict
            Maps dot-notation parameter keys to lists of values.
            Example: ``{"risk.stop_atr_mult": [1.0, 1.5, 2.0]}``
        df : pd.DataFrame, optional
            Pre-loaded data (avoids re-reading CSV for each run).

        Returns
        -------
        list of BacktestResult
            Sorted by Sharpe ratio descending.
        """
        if df is None:
            feed = CSVFeed(self.feed_path)
            df = feed.load()

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        logger.info("Running %d parameter combinations", len(combos))
        results = []

        for i, combo in enumerate(combos):
            overrides = dict(zip(keys, combo))
            logger.info("  [%d/%d] %s", i + 1, len(combos), overrides)
            result = self.run(df=df, config_overrides=overrides)
            results.append(result)

        # Sort by Sharpe descending
        results.sort(key=lambda r: r.sharpe, reverse=True)
        return results
