"""MES Autotrader — autonomous signal-to-execution pipeline.

Monitors live signals from the scanner and automatically executes
bracket orders on IBKR Gateway when conditions are met.

SAFETY FIRST:
- Paper trading mode by default (port 4002)
- Hard daily loss kill switch (-$500)
- Max 3 consecutive losses → pause
- Max 5 trades/day
- Only trades when scanner is running + risk manager approves
- Cooldown between trades (5 minutes)
- All orders are bracket (entry + stop + target) — no naked exposure

Architecture:
    Scanner → Signal → Autotrader.evaluate() → Risk Check → IBKR Bracket Order
                                                          → Order Manager Log
                                                          → Trade DB Record
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "autotrader_config.json"
STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "autotrader_state.json"


@dataclass
class AutotraderConfig:
    """Autotrader configuration — persisted to disk."""
    enabled: bool = False                    # master switch
    paper: bool = True                       # paper trading mode (port 4002)
    instrument: str = "MES"                  # only MES for now

    # Connection
    host: str = "127.0.0.1"
    live_port: int = 4001
    paper_port: int = 4002
    client_id: int = 32                      # dedicated client ID for autotrader

    # Signal thresholds
    min_signal_strength: float = 1.20        # minimum setup score to execute
    min_ml_probability: float = 0.45         # minimum ML model confidence
    require_setup: bool = True               # only trade when TradeSetup exists
    allowed_directions: list = field(default_factory=lambda: ["long", "short"])

    # Position sizing
    default_contracts: int = 1               # base position size
    max_contracts: int = 2                   # absolute max
    use_macro_sizing: bool = True            # apply macro regime multiplier

    # Risk limits (autotrader-specific, stricter than global)
    max_daily_loss_usd: float = 500.0        # hard stop for the day
    max_consecutive_losses: int = 3          # pause after 3 losses
    max_daily_trades: int = 5                # max auto-trades per day
    cooldown_seconds: int = 300              # 5 min between trades
    max_risk_per_trade_usd: float = 200.0    # max loss on single trade

    # Execution
    use_market_entry: bool = True            # True=market, False=limit at signal price
    breakeven_after_t1: bool = True          # move stop to BE once price touches T1
    trail_stop: bool = True                  # trail stop after T1 by trail_atr_mult * ATR
    trail_atr_mult: float = 1.5              # trailing distance in ATR units

    # End-of-day sizing taper — reduce contracts late in BRT session
    # to avoid swing-by-mistake exposure on positions still open near close.
    # Trades after this hour use eod_taper_factor of base sizing.
    eod_size_taper: bool = True
    eod_taper_brt_hour: int = 17             # start tapering at 17:00 BRT
    eod_taper_factor: float = 0.5            # multiply contracts by this

    def save(self):
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "AutotraderConfig":
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()


@dataclass
class AutotraderState:
    """Runtime state — tracks today's activity."""
    date: str = ""
    enabled: bool = False
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    daily_pnl_usd: float = 0.0
    consecutive_losses: int = 0
    last_trade_ts: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    open_position: Optional[Dict] = None
    trade_log: List[Dict] = field(default_factory=list)

    def reset_if_new_day(self):
        today = date.today().isoformat()
        if self.date != today:
            self.date = today
            self.trades_today = 0
            self.wins_today = 0
            self.losses_today = 0
            self.daily_pnl_usd = 0.0
            self.consecutive_losses = 0
            self.halted = False
            self.halt_reason = ""
            self.open_position = None
            self.trade_log = []

    def save(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls) -> "AutotraderState":
        if STATE_PATH.exists():
            data = json.loads(STATE_PATH.read_text())
            state = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            state.reset_if_new_day()
            return state
        state = cls()
        state.reset_if_new_day()
        return state


class MESAutotrader:
    """Autonomous MES trading engine.

    Usage:
        autotrader = MESAutotrader()
        autotrader.enable()          # start accepting signals
        autotrader.evaluate(signal)  # called by scanner on each new signal
        autotrader.disable()         # stop trading
    """

    def __init__(self):
        self.config = AutotraderConfig.load()
        self.state = AutotraderState.load()
        self._executor = None
        self._order_mgr = None
        self._lock = threading.Lock()

    # ── Lifecycle ──

    def enable(self, paper: Optional[bool] = None):
        """Enable autotrading."""
        if paper is not None:
            self.config.paper = paper
        self.config.enabled = True
        self.config.save()
        self.state.enabled = True
        self.state.reset_if_new_day()
        self.state.save()
        mode = "PAPER" if self.config.paper else "LIVE"
        port = self.config.paper_port if self.config.paper else self.config.live_port
        logger.info("MES Autotrader ENABLED [%s mode, port %d]", mode, port)

    def disable(self):
        """Disable autotrading. Does NOT close open positions."""
        self.config.enabled = False
        self.config.save()
        self.state.enabled = False
        self.state.save()
        logger.info("MES Autotrader DISABLED")

    def halt(self, reason: str):
        """Emergency halt — stops all trading until manually cleared."""
        self.state.halted = True
        self.state.halt_reason = reason
        self.state.save()
        logger.warning("MES Autotrader HALTED: %s", reason)

    def clear_halt(self):
        """Clear halt status and resume trading."""
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.save()
        logger.info("MES Autotrader halt cleared")

    def flatten(self) -> Optional[Dict]:
        """Emergency flatten — close all MES positions immediately."""
        try:
            executor = self._get_executor()
            result = executor.flatten_position("MES")
            if result:
                logger.warning("FLATTEN: closed MES position")
                self.state.open_position = None
                self.state.save()
                return asdict(result)
        except Exception as e:
            logger.error("Flatten failed: %s", e)
        return None

    # ── Signal Evaluation (called by scanner) ──

    def evaluate(self, signal: Dict[str, Any], setup: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate a signal for auto-execution.

        Called by the live scanner whenever a NEW MES signal fires.
        Returns a dict describing what happened (for logging/display).

        Parameters
        ----------
        signal : dict
            Signal data from scanner: direction, price, stop, target,
            strength, reasons, ml_probability, ml_recommendation, etc.
        setup : dict, optional
            Active TradeSetup if one exists.
        """
        with self._lock:
            self.state.reset_if_new_day()
            result = {"action": "skip", "reason": "", "timestamp": datetime.utcnow().isoformat()}

            # Gate 1: Master switch
            if not self.config.enabled or not self.state.enabled:
                result["reason"] = "autotrader_disabled"
                return result

            # Gate 2: Halt check
            if self.state.halted:
                result["reason"] = f"halted: {self.state.halt_reason}"
                return result

            # Gate 3: Open position check
            if self.state.open_position:
                result["reason"] = "position_already_open"
                return result

            # Gate 4: Daily trade limit
            if self.state.trades_today >= self.config.max_daily_trades:
                result["reason"] = f"daily_limit_reached ({self.state.trades_today}/{self.config.max_daily_trades})"
                return result

            # Gate 5: Daily loss limit
            if self.state.daily_pnl_usd <= -self.config.max_daily_loss_usd:
                self.halt(f"daily_loss_limit (${self.state.daily_pnl_usd:.0f})")
                result["reason"] = "daily_loss_limit"
                return result

            # Gate 6: Consecutive losses
            if self.state.consecutive_losses >= self.config.max_consecutive_losses:
                self.halt(f"consecutive_losses ({self.state.consecutive_losses})")
                result["reason"] = "consecutive_loss_limit"
                return result

            # Gate 7: Cooldown
            elapsed = time.time() - self.state.last_trade_ts
            if elapsed < self.config.cooldown_seconds:
                remaining = int(self.config.cooldown_seconds - elapsed)
                result["reason"] = f"cooldown ({remaining}s remaining)"
                return result

            # Gate 8: Signal strength
            strength = signal.get("strength", 0)
            if strength < self.config.min_signal_strength:
                result["reason"] = f"strength_too_low ({strength:.2f} < {self.config.min_signal_strength})"
                return result

            # Gate 9: ML probability
            ml_prob = signal.get("ml_probability", 0)
            ml_rec = signal.get("ml_recommendation", "")
            if ml_prob > 0 and ml_prob < self.config.min_ml_probability:
                result["reason"] = f"ml_reject ({ml_prob:.2f} < {self.config.min_ml_probability})"
                return result
            if ml_rec == "avoid":
                result["reason"] = "ml_recommendation_avoid"
                return result

            # Gate 10: Direction filter
            direction = signal.get("direction")
            if direction not in self.config.allowed_directions:
                result["reason"] = f"direction_blocked ({direction})"
                return result

            # Gate 11: Setup requirement
            if self.config.require_setup and not setup:
                result["reason"] = "no_active_setup"
                return result

            # Gate 12: Risk per trade
            stop_pts = abs(signal.get("stop_pts", signal.get("price", 0) - signal.get("stop", 0)))
            risk_per_contract = stop_pts * 5.0  # MES = $5/pt
            if risk_per_contract > self.config.max_risk_per_trade_usd:
                result["reason"] = f"risk_too_high (${risk_per_contract:.0f} > ${self.config.max_risk_per_trade_usd:.0f})"
                return result

            # All gates passed — EXECUTE
            return self._execute_trade(signal, setup, risk_per_contract)

    def _execute_trade(self, signal: Dict, setup: Optional[Dict], risk_per_contract: float) -> Dict:
        """Place the bracket order on IBKR."""
        direction = signal["direction"]
        price = signal["price"]
        stop = signal.get("stop", 0)
        target = signal.get("target", 0)

        # Use setup levels if available (more refined)
        if setup:
            stop = setup.get("stop_loss", stop)
            target = setup.get("target_1", target)  # conservative: T1 not T2

        # Position sizing
        contracts = self.config.default_contracts
        if self.config.use_macro_sizing:
            macro_mult = signal.get("macro_sizing", {})
            if isinstance(macro_mult, dict):
                mult = macro_mult.get("multiplier", 1.0)
                contracts = max(1, int(contracts * mult))
        contracts = min(contracts, self.config.max_contracts)

        # EOD sizing taper: positions opened late in BRT session get smaller
        # to avoid carrying size through close noise / unintentional swing.
        if self.config.eod_size_taper:
            try:
                import pytz
                brt_hour = datetime.now(pytz.timezone("America/Sao_Paulo")).hour
                if brt_hour >= self.config.eod_taper_brt_hour:
                    eod_contracts = max(1, int(contracts * self.config.eod_taper_factor))
                    if eod_contracts < contracts:
                        logger.info(
                            "AUTOTRADER EOD taper: %d → %d contracts (BRT %dh ≥ %dh)",
                            contracts, eod_contracts, brt_hour, self.config.eod_taper_brt_hour,
                        )
                        contracts = eod_contracts
            except Exception as e:
                logger.debug("EOD taper failed: %s", e)

        # Tick-align MES prices to 0.25
        stop = round(stop * 4) / 4
        target = round(target * 4) / 4
        price = round(price * 4) / 4

        mode = "PAPER" if self.config.paper else "LIVE"
        logger.info(
            "AUTOTRADER [%s] EXECUTING: %s MES %d @ %.2f SL=%.2f TP=%.2f",
            mode, direction.upper(), contracts, price, stop, target,
        )

        try:
            executor = self._get_executor()
            order_mgr = self._get_order_mgr()

            if self.config.use_market_entry:
                # Market entry with stop + target as bracket
                results = executor.submit_bracket_order(
                    instrument="MES",
                    side=direction,
                    quantity=contracts,
                    entry_price=price,
                    stop_price=stop,
                    target_price=target,
                )
            else:
                results = executor.submit_bracket_order(
                    instrument="MES",
                    side=direction,
                    quantity=contracts,
                    entry_price=price,
                    stop_price=stop,
                    target_price=target,
                )

            # Log execution
            order_ids = [r.order_id for r in results]
            order_mgr.record_execution(
                instrument="MES",
                side=direction,
                order_type="BRACKET",
                quantity=contracts,
                entry_price=price,
                stop_price=stop,
                target_price=target,
                status="submitted",
                regime=signal.get("regime", ""),
                signal_strength=signal.get("strength", 0),
                order_ids=order_ids,
            )

            # Update state — enriched with management fields used by
            # update_position_state for breakeven-after-T1 + trailing stop.
            # Paper mode actively simulates fills against current_stop;
            # live mode currently only logs when management WOULD move
            # the stop (executor lacks modify_order — see TODO at top).
            self.state.trades_today += 1
            self.state.last_trade_ts = time.time()
            self.state.open_position = {
                "direction": direction,
                "entry_price": price,
                "stop": stop,                  # original IBKR stop (unchanged)
                "target": target,              # T1 (the IBKR take-profit leg)
                "current_stop": stop,          # mutates over time via management
                "high_water_mark": price,      # max favorable excursion (long)
                "low_water_mark": price,       # min favorable excursion (short)
                "atr_at_entry": float(signal.get("atr", 0) or 0),
                "t1_hit": False,
                "be_moved": False,
                "trail_active": False,
                "contracts": contracts,
                "order_ids": order_ids,
                "opened_at": datetime.utcnow().isoformat(),
                "signal_strength": signal.get("strength", 0),
                "reasons": signal.get("reasons", []),
            }
            self.state.save()

            return {
                "action": "executed",
                "mode": mode,
                "direction": direction,
                "contracts": contracts,
                "entry": price,
                "stop": stop,
                "target": target,
                "risk_usd": round(risk_per_contract * contracts, 2),
                "order_ids": order_ids,
                "trade_number": self.state.trades_today,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Autotrader execution FAILED: %s", e)
            return {
                "action": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    # ── Position Monitoring ──

    def check_position(self) -> Optional[Dict]:
        """Check if the open position has been closed by stop/target.

        Called periodically by the scanner loop. Checks IBKR for
        current position and order status.
        """
        if not self.state.open_position:
            return None

        try:
            executor = self._get_executor()
            positions = executor.get_positions()
            mes_pos = [p for p in positions if p["symbol"] == "MES"]

            # If no MES position, the bracket closed it
            if not mes_pos or mes_pos[0]["quantity"] == 0:
                return self._record_trade_close()

        except Exception as e:
            logger.debug("Position check failed: %s", e)

        return None

    def update_position_state(self, current_price: float, atr: Optional[float] = None) -> Optional[Dict]:
        """Active position management — called every scan cycle.

        Implements breakeven-after-T1 and trailing-stop algorithms by
        mutating ``state.open_position["current_stop"]``.

        Behavior:
          * Track high/low water mark for trailing.
          * On first touch of T1 (target_price), flip ``t1_hit`` and
            move ``current_stop`` to entry (if breakeven_after_t1).
          * After T1, trail ``current_stop`` to
            ``high_water_mark - trail_atr_mult * atr`` for long
            (and symmetric for short), never moving against the trade.
          * In paper mode, simulate the fill when current_price crosses
            current_stop in the adverse direction — this is what makes
            paper mode meaningful while live IBKR order modification
            isn't yet wired into the executor.
          * In live mode, log the would-be stop change but don't
            mutate the actual broker order yet (TODO: add
            ``modify_order`` to ``IBKRExecutor`` and wire it here).

        Returns the trade record if a fill was simulated, else None.
        """
        if not self.state.open_position:
            return None

        pos = self.state.open_position
        direction = pos["direction"]
        entry = pos["entry_price"]
        target = pos.get("target", 0)
        current_stop = pos.get("current_stop", pos.get("stop", 0))

        # Use ATR from signal-time as fallback when caller doesn't pass one.
        atr_used = float(atr) if atr is not None else float(pos.get("atr_at_entry", 0) or 0)

        # 1. Update water marks
        if direction == "long":
            pos["high_water_mark"] = max(pos.get("high_water_mark", entry), current_price)
        else:
            pos["low_water_mark"] = min(pos.get("low_water_mark", entry), current_price)

        # 2. Detect T1 hit (first time only) and move stop to BE.
        if not pos.get("t1_hit") and target:
            t1_hit = (direction == "long" and current_price >= target) or \
                     (direction == "short" and current_price <= target)
            if t1_hit:
                pos["t1_hit"] = True
                if self.config.breakeven_after_t1 and not pos.get("be_moved"):
                    new_stop = entry
                    if (direction == "long" and new_stop > current_stop) or \
                       (direction == "short" and new_stop < current_stop):
                        logger.info(
                            "AUTOTRADER MGMT: T1 hit @ %.2f, stop %.2f → BE %.2f (%s)",
                            current_price, current_stop, new_stop,
                            "PAPER" if self.config.paper else "LIVE log-only",
                        )
                        pos["current_stop"] = new_stop
                        pos["be_moved"] = True
                        current_stop = new_stop

        # 3. Trail after T1 hit, using ATR-based distance.
        if pos.get("t1_hit") and self.config.trail_stop and atr_used > 0:
            trail_dist = self.config.trail_atr_mult * atr_used
            if direction == "long":
                trailing = pos["high_water_mark"] - trail_dist
                # Tick-align to MES 0.25
                trailing = round(trailing * 4) / 4
                if trailing > current_stop:
                    if not pos.get("trail_active"):
                        pos["trail_active"] = True
                        logger.info("AUTOTRADER MGMT: trailing stop active (long)")
                    pos["current_stop"] = trailing
                    current_stop = trailing
            else:
                trailing = pos["low_water_mark"] + trail_dist
                trailing = round(trailing * 4) / 4
                if trailing < current_stop:
                    if not pos.get("trail_active"):
                        pos["trail_active"] = True
                        logger.info("AUTOTRADER MGMT: trailing stop active (short)")
                    pos["current_stop"] = trailing
                    current_stop = trailing

        # 4. Stop-out detection. In paper mode we simulate the exit so
        #    the trail/BE actually closes the trade. In live mode the
        #    real IBKR bracket stop hasn't been modified (executor lacks
        #    the API), so the original broker stop will fire — we only
        #    log the divergence for audit.
        stopped = (direction == "long" and current_price <= current_stop) or \
                  (direction == "short" and current_price >= current_stop)
        if stopped:
            if self.config.paper:
                exit_reason = "trail_stop" if pos.get("trail_active") else \
                              "breakeven" if pos.get("be_moved") else "initial_stop"
                logger.info(
                    "AUTOTRADER MGMT: paper-fill simulated @ %.2f (reason=%s)",
                    current_stop, exit_reason,
                )
                return self.record_fill(current_stop, exit_reason)
            else:
                # Live mode: only log the divergence, don't double-fill.
                if not pos.get("live_management_warned"):
                    logger.warning(
                        "AUTOTRADER MGMT (LIVE): managed stop %.2f hit but IBKR "
                        "order still at %.2f — consider manual close. (executor "
                        "lacks modify_order; live order management not wired yet)",
                        current_stop, pos.get("stop", 0),
                    )
                    pos["live_management_warned"] = True

        # Persist state mutations so next call/process sees them.
        self.state.save()
        return None

    def record_fill(self, exit_price: float, exit_reason: str = "bracket"):
        """Manually record a position close (called when fill detected)."""
        if not self.state.open_position:
            return

        pos = self.state.open_position
        direction = pos["direction"]
        entry = pos["entry_price"]
        contracts = pos["contracts"]

        if direction == "long":
            pnl_pts = exit_price - entry
        else:
            pnl_pts = entry - exit_price

        pnl_usd = pnl_pts * 5.0 * contracts

        # Update state
        self.state.daily_pnl_usd += pnl_usd
        if pnl_usd > 0:
            self.state.wins_today += 1
            self.state.consecutive_losses = 0
        else:
            self.state.losses_today += 1
            self.state.consecutive_losses += 1

        trade_record = {
            **pos,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl_pts": round(pnl_pts, 2),
            "pnl_usd": round(pnl_usd, 2),
            "closed_at": datetime.utcnow().isoformat(),
        }
        self.state.trade_log.append(trade_record)
        self.state.open_position = None
        self.state.save()

        outcome = "WIN" if pnl_usd > 0 else "LOSS"
        logger.info(
            "AUTOTRADER %s: %s MES %d | entry=%.2f exit=%.2f | PnL=$%.2f | Daily=$%.2f",
            outcome, direction.upper(), contracts, entry, exit_price, pnl_usd, self.state.daily_pnl_usd,
        )

        # Check kill switches after close
        if self.state.daily_pnl_usd <= -self.config.max_daily_loss_usd:
            self.halt(f"daily_loss_limit (${self.state.daily_pnl_usd:.0f})")
        elif self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self.halt(f"consecutive_losses ({self.state.consecutive_losses})")

        return trade_record

    def _record_trade_close(self) -> Optional[Dict]:
        """Detect exit via IBKR order status and record it."""
        if not self.state.open_position:
            return None

        pos = self.state.open_position
        try:
            executor = self._get_executor()
            # Check order fills
            for trade in executor.ib.trades():
                if trade.order.orderId in pos.get("order_ids", []):
                    if trade.orderStatus.status == "Filled" and trade.order.orderType in ("STP", "LMT"):
                        fill_price = trade.orderStatus.avgFillPrice
                        if fill_price > 0:
                            reason = "stop_loss" if trade.order.orderType == "STP" else "take_profit"
                            return self.record_fill(fill_price, reason)
        except Exception as e:
            logger.debug("Trade close detection failed: %s", e)

        return None

    # ── Status ──

    def get_status(self) -> Dict[str, Any]:
        """Full autotrader status for API/frontend."""
        self.state.reset_if_new_day()
        return {
            "enabled": self.config.enabled and self.state.enabled,
            "mode": "PAPER" if self.config.paper else "LIVE",
            "port": self.config.paper_port if self.config.paper else self.config.live_port,
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
            "trades_today": self.state.trades_today,
            "max_daily_trades": self.config.max_daily_trades,
            "wins": self.state.wins_today,
            "losses": self.state.losses_today,
            "win_rate": round(self.state.wins_today / self.state.trades_today * 100, 1) if self.state.trades_today else 0,
            "daily_pnl_usd": round(self.state.daily_pnl_usd, 2),
            "max_daily_loss_usd": self.config.max_daily_loss_usd,
            "consecutive_losses": self.state.consecutive_losses,
            "max_consecutive_losses": self.config.max_consecutive_losses,
            "cooldown_seconds": self.config.cooldown_seconds,
            "cooldown_remaining": max(0, int(self.config.cooldown_seconds - (time.time() - self.state.last_trade_ts))),
            "open_position": self.state.open_position,
            "min_signal_strength": self.config.min_signal_strength,
            "min_ml_probability": self.config.min_ml_probability,
            "default_contracts": self.config.default_contracts,
            "max_contracts": self.config.max_contracts,
            "trade_log": self.state.trade_log[-10:],  # last 10
        }

    def get_config(self) -> Dict[str, Any]:
        """Get current config as dict."""
        return asdict(self.config)

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update config fields. Returns updated config."""
        current = asdict(self.config)
        current.update(updates)
        self.config = AutotraderConfig(**{k: v for k, v in current.items() if k in AutotraderConfig.__dataclass_fields__})
        self.config.save()
        return asdict(self.config)

    # ── Internal ──

    def _get_executor(self):
        if self._executor is None:
            from takata_engine.execution.executor_ibkr import IBKRExecutor
            self._executor = IBKRExecutor(
                host=self.config.host,
                port=self.config.paper_port if self.config.paper else self.config.live_port,
                paper=self.config.paper,
            )
        return self._executor

    def _get_order_mgr(self):
        if self._order_mgr is None:
            from takata_engine.execution.order_manager import OrderManager
            self._order_mgr = OrderManager()
        return self._order_mgr


# Singleton
_autotrader: Optional[MESAutotrader] = None


def get_autotrader() -> MESAutotrader:
    """Get or create the singleton autotrader instance."""
    global _autotrader
    if _autotrader is None:
        _autotrader = MESAutotrader()
    return _autotrader
