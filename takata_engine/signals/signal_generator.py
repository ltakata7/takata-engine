"""Signal generator — combines indicators + regime context into trade signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from takata_engine.indicators import ema, macd, rsi, bollinger, adx, atr, vwap
from takata_engine.regime import RegimeDetector, RegimeParams
from takata_engine.signals.filters import SignalFilter
from takata_engine.signals.position import Position, PositionManager
from takata_engine.signals.risk_manager import RiskManager
from takata_engine.utils.logger import TradeLogger

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A generated trade signal."""

    timestamp: datetime
    instrument: str
    direction: str  # "long", "short", "exit"
    price: float
    stop: float
    target: float
    regime: str
    strength: float  # 0.0–1.0 composite
    reasons: List[str] = field(default_factory=list)
    indicators: Dict[str, float] = field(default_factory=dict)


class SignalGenerator:
    """Generates trade signals by combining indicator conditions with regime context.

    This is the central orchestrator. Each bar:
    1. Detect regime → load adapted parameters
    2. Compute indicators with adapted parameters
    3. Evaluate entry/exit conditions
    4. Apply filters (session, etc.)
    5. Manage positions (open/close)
    6. Log everything

    Parameters
    ----------
    config : dict
        Full instrument config.
    regime_detector : RegimeDetector
        Fitted regime detector.
    trade_logger : TradeLogger, optional
        Logger for signals and trades.
    account_risk_per_trade : float
        Max loss per trade in currency units. Default 100.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        regime_detector: RegimeDetector,
        trade_logger: Optional[TradeLogger] = None,
        account_risk_per_trade: float = 100.0,
    ):
        self.config = config
        self.instrument = config["instrument"]["symbol"]
        self.point_value = config["instrument"]["point_value"]
        self.regime_detector = regime_detector
        self.regime_params = RegimeParams(config)
        self.trade_logger = trade_logger
        self.account_risk_per_trade = account_risk_per_trade

        self.signal_filter = SignalFilter.from_config(self.instrument, config)
        self.risk_mgr = RiskManager(config["risk"], self.point_value)
        self.position_mgr = PositionManager(self.point_value)

        self._prev_rsi: Optional[float] = None
        self._prev_macd_hist: Optional[float] = None

    def _compute_indicators(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute all indicators and return the latest values as a dict."""
        ind = params["indicators"]
        close = df["close"]
        latest = len(df) - 1

        ema_fast = ema(close, ind["ema"]["fast"])
        ema_slow = ema(close, ind["ema"]["slow"])
        macd_df = macd(close, ind["macd"]["fast"], ind["macd"]["slow"], ind["macd"]["signal"])
        rsi_s = rsi(close, ind["rsi"]["period"])
        bb = bollinger(close, ind["bollinger"]["period"], ind["bollinger"]["std_dev"])
        adx_df = adx(df, ind["adx"]["period"])
        atr_s = atr(df, ind["atr"]["period"])
        vwap_s = vwap(df)

        return {
            "ema_fast": ema_fast.iloc[latest],
            "ema_slow": ema_slow.iloc[latest],
            "macd": macd_df["macd"].iloc[latest],
            "macd_signal": macd_df["signal"].iloc[latest],
            "macd_hist": macd_df["histogram"].iloc[latest],
            "rsi": rsi_s.iloc[latest],
            "bb_upper": bb["upper"].iloc[latest],
            "bb_middle": bb["middle"].iloc[latest],
            "bb_lower": bb["lower"].iloc[latest],
            "adx": adx_df["adx"].iloc[latest],
            "plus_di": adx_df["plus_di"].iloc[latest],
            "minus_di": adx_df["minus_di"].iloc[latest],
            "atr": atr_s.iloc[latest],
            "vwap": vwap_s.iloc[latest],
            "close": close.iloc[latest],
            # Thresholds from config
            "_overbought": ind["rsi"]["overbought"],
            "_oversold": ind["rsi"]["oversold"],
            "_adx_threshold": ind["adx"]["trend_threshold"],
        }

    def _evaluate_long(self, ind: Dict[str, float]) -> tuple:
        """Evaluate long entry conditions. Returns (score, reasons)."""
        reasons = []
        score = 0.0

        # Trend: EMA fast > EMA slow
        if ind["ema_fast"] > ind["ema_slow"]:
            reasons.append("ema_bullish_cross")
            score += 0.25

        # Momentum: RSI not overbought and showing strength
        rsi_val = ind["rsi"]
        if not np.isnan(rsi_val):
            if ind["_oversold"] < rsi_val < ind["_overbought"]:
                if self._prev_rsi is not None and rsi_val > self._prev_rsi:
                    reasons.append("rsi_rising")
                    score += 0.15
            if self._prev_rsi is not None and self._prev_rsi <= ind["_oversold"] < rsi_val:
                reasons.append("rsi_oversold_cross")
                score += 0.20

        # MACD confirmation
        if ind["macd_hist"] > 0:
            reasons.append("macd_positive")
            score += 0.15
        elif self._prev_macd_hist is not None and self._prev_macd_hist <= 0 < ind["macd_hist"]:
            reasons.append("macd_cross_up")
            score += 0.20

        # Value: price above VWAP
        if ind["close"] > ind["vwap"]:
            reasons.append("above_vwap")
            score += 0.15

        # Trend strength: ADX above threshold
        if not np.isnan(ind["adx"]) and ind["adx"] > ind["_adx_threshold"]:
            if ind["plus_di"] > ind["minus_di"]:
                reasons.append("adx_strong_uptrend")
                score += 0.10

        return min(score, 1.0), reasons

    def _evaluate_short(self, ind: Dict[str, float]) -> tuple:
        """Evaluate short entry conditions. Returns (score, reasons)."""
        reasons = []
        score = 0.0

        # Trend: EMA fast < EMA slow
        if ind["ema_fast"] < ind["ema_slow"]:
            reasons.append("ema_bearish_cross")
            score += 0.25

        # Momentum: RSI showing weakness
        rsi_val = ind["rsi"]
        if not np.isnan(rsi_val):
            if ind["_oversold"] < rsi_val < ind["_overbought"]:
                if self._prev_rsi is not None and rsi_val < self._prev_rsi:
                    reasons.append("rsi_falling")
                    score += 0.15
            if self._prev_rsi is not None and self._prev_rsi >= ind["_overbought"] > rsi_val:
                reasons.append("rsi_overbought_cross")
                score += 0.20

        # MACD confirmation
        if ind["macd_hist"] < 0:
            reasons.append("macd_negative")
            score += 0.15
        elif self._prev_macd_hist is not None and self._prev_macd_hist >= 0 > ind["macd_hist"]:
            reasons.append("macd_cross_down")
            score += 0.20

        # Value: price below VWAP
        if ind["close"] < ind["vwap"]:
            reasons.append("below_vwap")
            score += 0.15

        # Trend strength: ADX above threshold
        if not np.isnan(ind["adx"]) and ind["adx"] > ind["_adx_threshold"]:
            if ind["minus_di"] > ind["plus_di"]:
                reasons.append("adx_strong_downtrend")
                score += 0.10

        return min(score, 1.0), reasons

    def _should_exit(self, position: Position, ind: Dict[str, float]) -> Optional[str]:
        """Check if an open position should be exited on signal logic (not stop/target)."""
        if position.side == "long":
            if ind["ema_fast"] < ind["ema_slow"]:
                return "ema_reversal"
        else:
            if ind["ema_fast"] > ind["ema_slow"]:
                return "ema_reversal"
        return None

    def run(self, df: pd.DataFrame, min_strength: float = 0.5) -> List[Signal]:
        """Run signal generation over a full DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with DatetimeIndex.
        min_strength : float
            Minimum composite score to trigger a signal.

        Returns
        -------
        list of Signal
            All generated signals.
        """
        # Need warmup bars for indicators
        warmup = 50
        if len(df) <= warmup:
            logger.warning("Not enough data for signal generation (need > %d bars)", warmup)
            return []

        signals: List[Signal] = []
        self._prev_rsi = None
        self._prev_macd_hist = None

        for i in range(warmup, len(df)):
            bar_time = df.index[i]
            window = df.iloc[: i + 1]
            bar = df.iloc[i]

            # Check session filter
            if not self.signal_filter.is_allowed(bar_time):
                # Still check exits for open positions
                closed = self.position_mgr.check_exits(self.instrument, bar, bar_time)
                if closed:
                    self._log_exit(closed)
                continue

            # Detect regime and load adapted params
            regime = self.regime_detector.detect(window)
            adapted = self.regime_params.get_params(regime)
            self.risk_mgr.update_from_regime(adapted["risk"])

            # Daily loss circuit breaker
            if not self.risk_mgr.check_daily_limit(
                self.position_mgr.daily_pnl(bar_time)
            ):
                logger.info("Daily loss limit reached at %s", bar_time)
                # Close any open position
                self.position_mgr.close_position(
                    self.instrument, bar["close"], bar_time, "daily_limit"
                )
                continue

            # Compute indicators with adapted parameters
            ind = self._compute_indicators(window, adapted)

            # Check stop/target for open position
            closed = self.position_mgr.check_exits(self.instrument, bar, bar_time)
            if closed:
                self._log_exit(closed)

            # Check signal-based exit
            open_pos = self.position_mgr.get_open(self.instrument)
            if open_pos:
                exit_reason = self._should_exit(open_pos, ind)
                if exit_reason:
                    closed = self.position_mgr.close_position(
                        self.instrument, bar["close"], bar_time, exit_reason
                    )
                    if closed:
                        self._log_exit(closed)

            # Generate new signals only if flat AND regime allows trading
            if not self.position_mgr.has_open(self.instrument):
                # Crisis regime with max_position_size=0 means NO TRADING
                if self.risk_mgr.max_position_size <= 0:
                    self._update_prev(ind)
                    continue

                long_score, long_reasons = self._evaluate_long(ind)
                short_score, short_reasons = self._evaluate_short(ind)

                atr_val = ind["atr"]
                if np.isnan(atr_val) or atr_val <= 0:
                    self._update_prev(ind)
                    continue

                signal = None
                if long_score >= min_strength and long_score > short_score:
                    stop = self.risk_mgr.compute_stop(bar["close"], atr_val, "long")
                    target = self.risk_mgr.compute_target(bar["close"], atr_val, "long")
                    size = self.risk_mgr.compute_size(
                        bar["close"], stop, self.account_risk_per_trade
                    )
                    if size > 0:
                        signal = Signal(
                            timestamp=bar_time,
                            instrument=self.instrument,
                            direction="long",
                            price=bar["close"],
                            stop=stop,
                            target=target,
                            regime=regime,
                            strength=long_score,
                            reasons=long_reasons,
                            indicators={k: v for k, v in ind.items() if not k.startswith("_")},
                        )

                elif short_score >= min_strength and short_score > long_score:
                    stop = self.risk_mgr.compute_stop(bar["close"], atr_val, "short")
                    target = self.risk_mgr.compute_target(bar["close"], atr_val, "short")
                    size = self.risk_mgr.compute_size(
                        bar["close"], stop, self.account_risk_per_trade
                    )
                    if size > 0:
                        signal = Signal(
                            timestamp=bar_time,
                            instrument=self.instrument,
                            direction="short",
                            price=bar["close"],
                            stop=stop,
                            target=target,
                            regime=regime,
                            strength=short_score,
                            reasons=short_reasons,
                            indicators={k: v for k, v in ind.items() if not k.startswith("_")},
                        )

                if signal:
                    signals.append(signal)
                    self._open_from_signal(signal, size, bar_time)

            self._update_prev(ind)

        # Force close any open position at end
        open_pos = self.position_mgr.get_open(self.instrument)
        if open_pos:
            closed = self.position_mgr.close_position(
                self.instrument, df["close"].iloc[-1], df.index[-1], "end_of_data"
            )
            if closed:
                self._log_exit(closed)

        return signals

    def _update_prev(self, ind: Dict[str, float]) -> None:
        self._prev_rsi = ind["rsi"]
        self._prev_macd_hist = ind["macd_hist"]

    def _open_from_signal(self, signal: Signal, size: float, time: datetime) -> None:
        self.position_mgr.open_position(
            instrument=signal.instrument,
            side=signal.direction,
            entry_price=signal.price,
            entry_time=time,
            size=size,
            stop=signal.stop,
            target=signal.target,
            regime=signal.regime,
        )
        if self.trade_logger:
            self.trade_logger.log_signal(
                instrument=signal.instrument,
                signal=signal.direction,
                price=signal.price,
                regime=signal.regime,
                indicators=signal.indicators,
                strength=signal.strength,
                reasons=signal.reasons,
            )
            self.trade_logger.log_entry(
                instrument=signal.instrument,
                side=signal.direction,
                price=signal.price,
                size=size,
                stop=signal.stop,
                target=signal.target,
                regime=signal.regime,
            )

    def _log_exit(self, pos: Position) -> None:
        if self.trade_logger:
            self.trade_logger.log_exit(
                instrument=pos.instrument,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=pos.exit_price,
                size=pos.size,
                reason=pos.exit_reason,
                regime=pos.regime,
            )

    def summary(self) -> str:
        """Return a printable trade summary."""
        positions = self.position_mgr.closed_positions
        if not positions:
            return "No trades generated."

        total_pnl = sum(p.pnl for p in positions)
        winners = [p for p in positions if p.pnl > 0]
        losers = [p for p in positions if p.pnl <= 0]
        win_rate = len(winners) / len(positions) * 100 if positions else 0

        avg_win = np.mean([p.pnl for p in winners]) if winners else 0
        avg_loss = np.mean([p.pnl for p in losers]) if losers else 0
        profit_factor = abs(sum(p.pnl for p in winners) / sum(p.pnl for p in losers)) if losers and sum(p.pnl for p in losers) != 0 else float("inf")

        # Regime breakdown
        regime_counts: Dict[str, int] = {}
        regime_pnl: Dict[str, float] = {}
        for p in positions:
            regime_counts[p.regime] = regime_counts.get(p.regime, 0) + 1
            regime_pnl[p.regime] = regime_pnl.get(p.regime, 0) + p.pnl

        # Exit reason breakdown
        exit_counts: Dict[str, int] = {}
        for p in positions:
            exit_counts[p.exit_reason] = exit_counts.get(p.exit_reason, 0) + 1

        lines = [
            "Trade Summary",
            "=" * 50,
            f"  Total trades:    {len(positions)}",
            f"  Winners:         {len(winners)} ({win_rate:.1f}%)",
            f"  Losers:          {len(losers)}",
            f"  Total P&L:       {total_pnl:,.2f}",
            f"  Avg win:         {avg_win:,.2f}",
            f"  Avg loss:        {avg_loss:,.2f}",
            f"  Profit factor:   {profit_factor:.2f}",
            "",
            "By Regime:",
        ]
        for regime in sorted(regime_counts):
            lines.append(
                f"  {regime:25s}: {regime_counts[regime]:3d} trades, "
                f"P&L={regime_pnl[regime]:,.2f}"
            )
        lines.append("")
        lines.append("By Exit Reason:")
        for reason in sorted(exit_counts):
            lines.append(f"  {reason:25s}: {exit_counts[reason]:3d}")

        return "\n".join(lines)
